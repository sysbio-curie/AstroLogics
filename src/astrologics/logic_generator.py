import mpbn
import bonesis
import sys
import os
import shutil
from tqdm.auto import tqdm
import multiprocessing as mp
import time
import re

def bonesis_ensemble_from_sif(file,
                              limit=1000,
                              exact_influence_graph=True,
                              extra_properties=lambda bo: bo,
                              diversity=True):
    """
    Build a Boolean network ensemble from a SIF-formatted influence graph.

    Args:
        file (str): Path to the SIF file describing signed interactions.
        limit (int): Maximum number of Boolean networks to sample.
        exact_influence_graph (bool): Whether to enforce an exact match of the influence graph.
        extra_properties (Callable[[bonesis.BoNesis], None]): Callable adding custom constraints to the solver.
        diversity (bool): If True, sample diverse networks; otherwise sample arbitrarily.

    Returns:
        list[mpbn.MPBooleanNetwork]: Generated Boolean network ensemble.
    """
    dom = bonesis.InfluenceGraph.from_sif(file, exact=exact_influence_graph)
    bo = bonesis.BoNesis(dom)

    # Add `extra` properties
    extra_properties(bo)
    if diversity:
        view = bo.diverse_boolean_networks(limit=limit)
    else:
        view = bo.boolean_networks(limit=limit)

    return list(view)


def bonesis_ensemble_from_single_bn(bn, limit=1000,
                    exact_influence_graph=True,
                    fixedpoints=None,
                    extra_properties=lambda bo: bo,
                    diversity=True,
                    maxclause=32,
                    **domain_opts):
    """
    Generate an ensemble of Boolean networks that respect the structure of a reference network.

    Args:
        bn (Union[str, mpbn.MPBooleanNetwork]): Reference Boolean network or path to a .bnet file.
        limit (int): Maximum number of Boolean networks to produce.
        exact_influence_graph (bool): Require generated networks to match the reference influence graph exactly.
        fixedpoints (Optional[str]): Strategy for constraining fixed points ("included", "same", or None).
        extra_properties (Callable[[bonesis.BoNesis], None]): Callable adding custom constraints to the solver.
        diversity (bool): If True, request diverse networks; otherwise allow duplicates.
        maxclause (int): Maximum number of clauses allowed per regulation in the influence graph.
        **domain_opts: Extra keyword arguments forwarded to bonesis.InfluenceGraph.

    Returns:
        list[mpbn.MPBooleanNetwork]: Boolean networks satisfying the imposed constraints.
    """

    bn = mpbn.MPBooleanNetwork.auto_cast(bn)
    dom = bonesis.InfluenceGraph(bn.influence_graph(),
                                 exact=exact_influence_graph,
                                 maxclause=maxclause, **domain_opts)

    data = {}
    if fixedpoints:
        for i, x in enumerate(bn.fixedpoints()):
            data[f"fp{i}"] = x

    bo = bonesis.BoNesis(dom, data)

    if fixedpoints in ["included", "same"]:
        for fp in data:
            bo.fixed(~bo.obs(fp))

    if fixedpoints == "same":
        bo.all_fixpoints({bo.obs(fp) for fp in data})

    extra_properties(bo)

    if diversity:
        view = bo.diverse_boolean_networks(limit=limit)
    else:
        view = bo.boolean_networks(limit=limit)

    return list(view)

def write_solution_file(index, solution, previous):
    """
    Persist a single Boolean network solution to disk.

    Args:
        index (int): Sequential identifier used to name the output file.
        solution (mpbn.MPBooleanNetwork): Boolean network instance to serialize.
        previous (syncmanager.ListProxy): Shared list tracking completed file indices.

    Returns:
        None
    """
    filename = f"bn_{index}.bnet"
    with open(filename, "w") as file:
        file.write(solution.source())
    previous.append(index)

def write_bn_files(solutions,
                   path,
                   project_name,
                   num_processes = 15):
    """
    Persist an ensemble of Boolean networks to .bnet files using multiprocessing.

    Args:
        solutions (Sequence[mpbn.MPBooleanNetwork]): Boolean networks to export.
        path (Optional[str]): Directory in which the project folder will be created.
        project_name (str): Name of the directory that will contain generated files.
        num_processes (int): Maximum number of concurrent writer processes.

    Returns:
        None
    """
    # Multiprocessing arg
    manager=mp.Manager()
    previous=manager.list()
    processes=[]
    
    # If path is defined
    if path is not None:
        os.chdir(path)
    
    # Make project directory (remove if exists)
    if os.path.exists(project_name):
        shutil.rmtree(project_name)
    os.mkdir(project_name)
    os.chdir(project_name)

    # For loop to write bnet files
    for i in tqdm(range(len(solutions))):
        solution = solutions[i]
        while len(previous)<i-(num_processes-1):
            time.sleep(1)
        p = mp.Process(target = write_solution_file, 
                        args = (i,solution,previous))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()