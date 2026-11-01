"""Deterministic operator message per vial, built only from the record's evidence. No model calls."""


def explain(r):
    vial, recipe, prev = r["vial"], r["recipe_id"], r.get("prev_recipe_id")
    if r.get("recipe_changed") == 1:
        context = f"first after recipe {prev}"
    elif r.get("recipe_changed") == 0:
        context = f"repeat {r['position_in_run']} of recipe {recipe}"
    else:
        context = "previous recipe unknown"
    head = f"Vial {vial} (recipe {recipe}, {context}): "
    if r["decision"] == "insufficient_evidence":
        return head + f"no decision; {r['reason']}. Inspect manually before release."
    d, tau = r["dE_vs_reference"], r["tau"]
    body = f"measured colour differs from recipe {recipe}'s settled colour by {d:.1f} (threshold {tau:.1f}). "
    if r["decision"] == "keep":
        return head + body + f"Within tolerance. Continue recipe {recipe}."
    if r.get("recipe_changed") == 1:
        cause = f"Likely carryover from recipe {prev} in the shared dispensing path. "
    else:
        cause = "Previous recipe was the same, so carryover is unlikely; check dispense volumes or the camera view. "
    return head + body + cause + f"Flush before continuing recipe {recipe}."
