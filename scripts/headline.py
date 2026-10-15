"""Headline pilot tables: CRD + OLS against each design analysed with its own RI."""

import pandas as pd
from summarize import ess, load, rates

DESIGNS = ["rerand", "exchange", "shell", "FSM"]

if __name__ == "__main__":
    d = load()
    d = d[d.p < d.N - 2]
    power = rates(d, null=False)
    size = rates(d, null=True)
    e = ess(d)
    rows = []
    for (n, p, dgp), g in power.groupby(level=["N", "p", "dgp"]):
        row = {"N": n, "p": p, "dgp": dgp, "CRD+OLS": g.loc[(n, p, dgp, "CRD"), "OLS"]}
        for des in DESIGNS:
            key = (n, p, dgp, des)
            row[f"{des} RI-OLS"] = g["RI OLS"].get(key)
            row[f"ESS {des}"] = e["ESS OLS vs CRD+OLS"].get(key)
        rows.append(row)
    out = pd.DataFrame(rows).set_index(["N", "p", "dgp"])
    pd.set_option("display.width", 220)
    print("POWER: CRD with OLS vs each design with RI on the OLS coefficient")
    print("ESS: MSE of CRD+OLS / MSE of design+OLS (above 1 favours the design)\n")
    print(out.round(2).to_string())
    worst = size[["RI diff", "RI OLS"]].stack()
    print(f"\nRI size across all cells: {worst.min():.3f} to {worst.max():.3f}")
