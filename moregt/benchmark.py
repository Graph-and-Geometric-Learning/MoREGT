"""The RelBench databases and the 21 entity tasks MoREGT is evaluated on."""
from __future__ import annotations

DATABASES = ("rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial")

#: Binary classification tasks, scored by TEST AUROC.
BINARY_TASKS = (
    ("rel-amazon", "user-churn"), ("rel-amazon", "item-churn"),
    ("rel-avito", "user-visits"), ("rel-avito", "user-clicks"),
    ("rel-event", "user-repeat"), ("rel-event", "user-ignore"),
    ("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"), ("rel-stack", "user-badge"),
    ("rel-trial", "study-outcome"),
)

#: Regression tasks, scored by TEST MAE normalised by the train target std.
REGRESSION_TASKS = (
    ("rel-amazon", "user-ltv"), ("rel-amazon", "item-ltv"),
    ("rel-avito", "ad-ctr"),
    ("rel-event", "user-attendance"),
    ("rel-f1", "driver-position"),
    ("rel-hm", "item-sales"),
    ("rel-stack", "post-votes"),
    ("rel-trial", "study-adverse"), ("rel-trial", "site-success"),
)

TASKS = BINARY_TASKS + REGRESSION_TASKS


def tasks_of(dataset: str) -> list[str]:
    return [t for ds, t in TASKS if ds == dataset]
