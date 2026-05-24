def simulate_day(states):
    """
    Симуляція денної стрес-батареї та ультрадіанних інтервалів.
    Одне вікно = 60 хвилин в поточній реалізації.
    """

    battery = 100.0
    timeline = []

    productive_streak = 0

    for i, state in enumerate(states):
        if state == "productive_stress":
            productive_streak += 1
        else:
            productive_streak = 0

        need_break = productive_streak >= 3

        if state == "overload":
            battery -= 3.0
        elif state == "productive_stress":
            battery -= 0.5
        elif state == "normal":
            battery -= 0.2

        if need_break:
            battery -= 2.0

        battery = max(0.0, min(100.0, battery))

        timeline.append({
            "step": i,
            "state": state,
            "battery": round(battery, 2),
            "need_break": need_break
        })

    return timeline