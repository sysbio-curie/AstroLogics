import bonesis
import pandas as pd
from bonesis0.asp_encoding import configurations_of_facts

class ConfigurationsView(bonesis.BonesisView):
    project = True
    show_templates = ["configuration"]

    def format_model(self, model):
        atoms = model.symbols(shown=True)
        return configurations_of_facts(atoms, keys="all")

def make_trajectory_v1(f, start, end, steps=None, ret_col=False, **kwargs):
    """
    f: BooleanNetwork
    start, end: configurations
    steps: number of steps to reach end from start (default=distance between start and end)
    """
    if steps is None:
        steps = len([n for n in f if start.get(n) != end.get(n)])
    bo = bonesis.BoNesis(f)
    traj = [~bo.obs(start)] + [bo.cfg() for _ in range(1, steps)] + [~bo.obs(end)]
    for x, y in zip(traj[:-1], traj[1:]):
        x >= y

    bo.custom(
        """
hsize(0,0).
hsize(X,K) :- K = #count { N: mcfg(X,N,1),mcfg(X,N,-1) }; mcfg(X,_,_).
maxdst(M) :- M = #max { K: hsize(_,K) }.
#minimize { K@1: maxdst(K) }.
"""
    )
    clingo_opts = ["--opt-strategy=bb"] + kwargs.pop("clingo_opts", [])

    sol = next(
        iter(ConfigurationsView(bo, mode="optN", clingo_opts=clingo_opts, **kwargs))
    )

    cfg_order = [c.name for c in traj]
    col_order = list(sorted(f))
    return pd.DataFrame.from_dict(sol, orient="index")[col_order].loc[cfg_order]


def labelled_trajectory(bn, start, stop, start_label, stop_label, steps=None, **kwargs
) -> pd.DataFrame:
    """
    Create a labeled trajectory using `bn_synthesis.trajectories.make_trajectory_v1`

    @param bn: A MostPermissive Boolean Network
    @param start: initial configuration of the trajectory.
    @param stop: last configuration of the trajectory.
    @param _trans_label: A label for intermediate configurations found by clingo.

    `steps` and `kwargs` are passed directly to `make_trajectory_v1`
    """
    traj_df = make_trajectory_v1(bn, start, stop, steps=steps, **kwargs)
    _trans_label = f"{start_label}_to_{stop_label}"
    new_index = traj_df.index.map(
        lambda x: f"{_trans_label}_{x.replace('__cfg', '')}" if "__cfg" in x else x
    ).to_list()
    new_index[0] = start_label
    new_index[-1] = stop_label

    return traj_df.set_index([new_index])
