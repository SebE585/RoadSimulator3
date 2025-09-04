# simulator/events/adapter.py
def inject_events_standard(df, config: dict, **cfg):
    from simulator.events.pipeline import apply_all_events
    # appeler direct avec la signature attendue par la fonction RS3
    return apply_all_events(df, config)