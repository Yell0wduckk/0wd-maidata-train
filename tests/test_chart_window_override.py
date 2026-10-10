"""Explicit target-window bounds have the same terminal semantics as stored bounds."""
import pytest

from maidata.chart import Grammar, Moment, Note, HoldDuration
from maidata.physical import Geometry, PhysicalState
from maidata.time import TimingMap


@pytest.mark.parametrize('stored_end, explicit_end, tick, expected_phase', [
    (None, 96, 96, 'closed'),
    (None, 96, 97, 'closed'),
    (None, 96, 95, 'note'),
    (192, 96, 96, 'closed'),
    (96, 192, 96, 'note'),
    (96, None, 96, 'closed'),
])
def test_grammar_explicit_window_end(stored_end, explicit_end, tick, expected_phase):
    timing = TimingMap([(0, 60)])
    grammar = Grammar(timing)
    grammar.inject_boundary(Moment(0), start_tick=0, window_end=stored_end)
    grammar.accept(Note.make('tap', 'A1'))
    grammar.accept(Moment(timing.delta_ms(0, tick)), window_end=explicit_end)
    assert grammar.state.phase == expected_phase
    assert grammar.state.last_real_tick == 0
    assert grammar.state.cursor == (0 if expected_phase == 'closed' else tick)


def test_physical_explicit_terminal_bypasses_note_completion_and_keeps_actions():
    timing = TimingMap([(0, 60)])
    state = PhysicalState(timing, Geometry({}), audio_end_seconds=20)
    state.inject_boundary(Moment(0), start_tick=0)
    for pad in ('A1', 'A2'):
        state.accept(Note.make('hold', pad))
        state.accept(HoldDuration(480))
    before = state.snapshot()
    terminal = Moment(timing.delta_ms(0, 96))
    assert not state.query(terminal).allowed
    assert state.query(terminal, window_end=96).allowed
    assert state.snapshot() == before
    state.accept(terminal, window_end=96)
    assert state.grammar.state.phase == 'closed'
    assert state.grammar.state.cursor == state.grammar.state.last_real_tick == 0
    assert state.actions == before.actions
    state.restore(before)
    assert state.snapshot() == before
