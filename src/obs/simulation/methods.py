"""The method pool: which agents can be run, and how they are drawn.

The four SAT-CTS variants are the SAME algorithm with two flags flipped
(`reset_priors`, `init_phase`), so they are coded by flag rather than given
four unrelated colours.
"""
from obs.algorithms.combinatorial import (
    CTSAgent, CUCBAgent, SATCTSUCBAgent, SATCTSv2SharedAgent,
)

NAMES = ["SAT-CTS", "SAT-CTS-Retain", "SAT-CTS-Init", "SAT-CTS-Retain-Init",
         "CTS", "CUCB", "CUCB-clip"]
NAMES_G = ["SAT-CTS-InitG2", "SAT-CTS-InitG5", "SAT-CTS-InitG10",
           "SAT-CTS-RetainInitG2", "SAT-CTS-RetainInitG5",
           "SAT-CTS-RetainInitG10"]

STYLE = {
    "SAT-CTS":             dict(color="#00B945", ls="-",  marker="o"),
    "SAT-CTS-Retain":      dict(color="#00B945", ls="--", marker="v"),
    "SAT-CTS-Init":        dict(color="#845B97", ls="-",  marker="P"),
    "SAT-CTS-Retain-Init": dict(color="#845B97", ls="--", marker="X"),
    "SAT-CTS-InitG2":      dict(color="#845B97", ls=":",  marker="p"),
    "SAT-CTS-InitG5":      dict(color="#C77CFF", ls=":",  marker="h"),
    "SAT-CTS-InitG10":     dict(color="#C77CFF", ls="-.", marker="*"),
    "SAT-CTS-RetainInitG2":  dict(color="#00B945", ls=":",  marker="p"),
    "SAT-CTS-RetainInitG5":  dict(color="#4DBF7A", ls=":",  marker="h"),
    "SAT-CTS-RetainInitG10": dict(color="#4DBF7A", ls="-.", marker="*"),
    "SAT-CTS-W":           dict(color="#FF2C00", ls="-",  marker="s"),
    "CTS":                 dict(color="#0C5DA5", ls="-.", marker="^"),
    "CUCB":                dict(color="#FF9500", ls=":",  marker="D"),
    "CUCB-clip":           dict(color="#E36414", ls=(0, (3, 1, 1, 1)), marker="v"),
}


def make_agents(methods, num_users, tb, rate_set, target, objective):
    """Fresh agent instances for one experiment.

    `objective` carries the per-BS RF-chain cap. CUCB is passed None instead --
    see UNCAPPED_METHODS in obs.config for why the exemption exists and why it
    can only help the baseline.
    """
    def sat(reset, init=False, group=None):
        kw = {} if group is None else {"init_group_size": group}
        return lambda: SATCTSv2SharedAgent(
            num_users, tb, rate_set, target, reset_priors=reset,
            init_phase=init, objective=objective, **kw)

    pool = {
        "SAT-CTS":              sat(True),
        "SAT-CTS-Retain":       sat(False),
        "SAT-CTS-Init":         sat(True, True),
        "SAT-CTS-Retain-Init":  sat(False, True),
        "SAT-CTS-InitG2":       sat(True, True, 2),
        "SAT-CTS-InitG5":       sat(True, True, 5),
        "SAT-CTS-InitG10":      sat(True, True, 10),
        "SAT-CTS-RetainInitG2":  sat(False, True, 2),
        "SAT-CTS-RetainInitG5":  sat(False, True, 5),
        "SAT-CTS-RetainInitG10": sat(False, True, 10),
        "SAT-CTS-W":      lambda: SATCTSUCBAgent(num_users, tb, rate_set, target,
                                                 objective=objective),
        "CTS":            lambda: CTSAgent(num_users, tb, rate_set,
                                           objective=objective),
        "CUCB":           lambda: CUCBAgent(num_users, tb, rate_set,
                                            objective=None),
        "CUCB-clip":      lambda: CUCBAgent(num_users, tb, rate_set,
                                            objective=None, clip=True),
    }
    bad = [m for m in methods if m not in pool]
    if bad:
        raise ValueError(f"unknown method(s) {bad}; known: {list(pool)}")
    return {m: pool[m]() for m in methods}
