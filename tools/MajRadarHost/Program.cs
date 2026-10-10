using MajRadar.Core;
using MajRadar.MajSimaiAdapter;
using MajRadar.Runtime;
using System.Text.Json;

// One full canonical chart per request. Adapt once; analyze its existing target
// windows independently. The pinned upstream analyzers and score mapping remain
// unchanged, including positive-duration empty-chart regression.
var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
var adapter = new MajSimaiChartAdapter();
var runtime = new RadarRuntime();
string? line;
while ((line = Console.ReadLine()) is not null)
{
    if (string.IsNullOrWhiteSpace(line)) continue;
    Request? request = null;
    try
    {
        request = JsonSerializer.Deserialize<Request>(line, json)
            ?? throw new InvalidOperationException("Missing request.");
        var adapted = await adapter.ParseAndAdaptAsync(request.Inote);
        if (!adapted.IsSuccess)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                request.RecordId, request.View, status = "error", errors = adapted.Errors,
                windows = Array.Empty<object>()
            }, json));
            continue;
        }
        var windows = new List<object>();
        foreach (var window in request.Windows)
        {
            var crop = Crop(adapted.Chart!, window);
            var result = runtime.Analyze(crop);
            windows.Add(new
            {
                window.SegmentId, status = result.Status, result.Errors,
                scores = result.Scores, raw_values = result.RawValues,
                objects = crop.Events.Count(item => item.Kind != RadarEventKind.Timing),
                chart_end_seconds = crop.ChartEndTimeSeconds,
                last_event_end_seconds = crop.LastEventEndTimeSeconds,
                effective_duration_seconds = Math.Max(crop.ChartEndTimeSeconds, crop.LastEventEndTimeSeconds ?? 0),
                result.ModelVersion, result.MappingVersion
            });
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            request.RecordId, request.View, status = "ok", errors = Array.Empty<string>(), windows
        }, json));
    }
    catch (Exception exception)
    {
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            record_id = request?.RecordId, view = request?.View, status = "error",
            errors = new[] { exception.Message }, windows = Array.Empty<object>()
        }, json));
    }
}

static RadarChartInput Crop(RadarChartInput full, Window window)
{
    if (window.EndTick - window.StartTick != 16 * 96 ||
        !double.IsFinite(window.StartSeconds) || !double.IsFinite(window.EndSeconds) ||
        window.EndSeconds <= window.StartSeconds)
        throw new InvalidOperationException("Expected a positive complete sixteen-beat target window.");
    var startBeat = new BeatPosition(window.StartTick, 96);
    var endBeat = new BeatPosition(window.EndTick, 96);
    // MajSimai stores BPM as float. Reconstruct both window boundaries from
    // its adapted clock, instead of subtracting the producer's double BPM
    // clock from native events (which makes exact boundary attacks negative).
    var timing = full.Events.Where(item => item.Kind == RadarEventKind.Timing).ToArray();
    double SecondsAt(BeatPosition position)
    {
        var point = timing.LastOrDefault(item => item.StartBeat <= position);
        if (point is null || point.Bpm is null)
            throw new InvalidOperationException("Window has no inherited BPM.");
        return point.StartTimeSeconds + (position - point.StartBeat).ToDouble() * 60 / point.Bpm.Value;
    }
    var startSeconds = SecondsAt(startBeat);
    var endSeconds = SecondsAt(endBeat);
    var owned = full.Events.Where(item => item.Kind != RadarEventKind.Timing &&
        (item.Kind == RadarEventKind.Slide ? item.SlideDeclareBeat ?? item.StartBeat : item.StartBeat) >= startBeat &&
        (item.Kind == RadarEventKind.Slide ? item.SlideDeclareBeat ?? item.StartBeat : item.StartBeat) < endBeat).ToArray();
    var lastEnd = owned.Length == 0 ? 0 : owned.Max(item => item.EndTimeSeconds) - startSeconds;
    var duration = endSeconds - startSeconds;
    var events = owned.Select(item => Rebase(item, startBeat, startSeconds)).ToList();
    var inherited = timing.LastOrDefault(item => item.StartBeat <= startBeat);
    if (inherited is null || inherited.Bpm is null)
        throw new InvalidOperationException("Window has no inherited BPM.");
    events.Add(new RadarEvent
    {
        EventId = 0, Kind = RadarEventKind.Timing, StartBeat = BeatPosition.Zero,
        EndBeat = BeatPosition.Zero, Bpm = inherited.Bpm
    });
    var effectiveEnd = startSeconds + Math.Max(duration, lastEnd);
    events.AddRange(timing.Where(item => item.StartBeat > startBeat &&
        item.StartTimeSeconds <= effectiveEnd + 1e-9).Select(item => Rebase(item, startBeat, startSeconds)));
    return new RadarChartInput
    {
        Events = events.OrderBy(item => item.StartTimeSeconds).ThenBy(item => item.DeclarationOrder).ThenBy(item => item.EventId).ToArray(),
        ChartEndTimeSeconds = duration, LastEventEndTimeSeconds = lastEnd
    };
}

static RadarEvent Rebase(RadarEvent item, BeatPosition beat, double seconds) => new()
{
    EventId = item.EventId, Kind = item.Kind, IsSlideHead = item.IsSlideHead,
    SlideDeclareTimeSeconds = item.SlideDeclareTimeSeconds is double declared ? declared - seconds : null,
    SlideDeclareBeat = item.SlideDeclareBeat is BeatPosition declaredBeat ? declaredBeat - beat : null,
    StartTimeSeconds = item.StartTimeSeconds - seconds, EndTimeSeconds = item.EndTimeSeconds - seconds,
    StartBeat = item.StartBeat - beat, EndBeat = item.EndBeat - beat,
    Bpm = item.Bpm, Position = item.Position, HeadEventId = item.HeadEventId,
    SlideGroupId = item.SlideGroupId, IsBreak = item.IsBreak, IsEx = item.IsEx,
    IsMine = item.IsMine, Flags = item.Flags, RawToken = item.RawToken,
    DeclarationOrder = item.DeclarationOrder,
    SlidePath = item.SlidePath?.Select(segment => new SlidePathSegment
    {
        Shape = segment.Shape, StartPosition = segment.StartPosition,
        ViaPosition = segment.ViaPosition, EndPosition = segment.EndPosition,
        BarCount = segment.BarCount, StartTimeSeconds = segment.StartTimeSeconds - seconds,
        EndTimeSeconds = segment.EndTimeSeconds - seconds, RawSegment = segment.RawSegment
    }).ToArray()
};

sealed class Request
{
    public string RecordId { get; set; } = "";
    public string View { get; set; } = "identity";
    public string Inote { get; set; } = "";
    public Window[] Windows { get; set; } = Array.Empty<Window>();
}

sealed class Window
{
    public string SegmentId { get; set; } = "";
    public int StartTick { get; set; }
    public int EndTick { get; set; }
    public double StartSeconds { get; set; }
    public double EndSeconds { get; set; }
}
