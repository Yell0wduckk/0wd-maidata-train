"""A failed expert returns to the router without committing its output row."""
from .reporting import failure_record, count
from maimai_baseline.physical.inference_state import PhysicalDecodeError


def select_with_router(outputs, *, choose_row, decode_again, has_router, **kw):
    excluded = set()
    while True:
        try:
            return choose_row(outputs, **kw)
        except PhysicalDecodeError as error:
            if not error.context.get('recoverable') or 'non-finite' in str(error):
                raise
            family = int(outputs['decoder_family'][0, -1]) if has_router else None
            failure_record(tick=kw['tick'], moment=kw['moment'], stage='family',
                           family=family, reason=str(error), detail=error.context)
            if not has_router:
                raise
            excluded.add(family)
            if len(excluded) == 2:
                raise
            # The selector applies grammar and physical support as well as these
            # exclusions. Failure here bubbles to the MOMENT transaction.
            try:
                outputs = decode_again(excluded)
            except PhysicalDecodeError as next_error:
                if not next_error.context.get('recoverable') or 'non-finite' in str(next_error):
                    raise
                raise error
            count('router_retry')
            failure_record(tick=kw['tick'], moment=kw['moment'], stage='router',
                           excluded_families=sorted(excluded),
                           selected_family=int(outputs['decoder_family'][0, -1]))
