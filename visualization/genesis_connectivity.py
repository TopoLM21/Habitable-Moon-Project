"""Shared labels for contact diagnostics, without importing plotting backends."""


def connectivity_text(diagnostics):
    """Old result files lack this distinction; do not invent missing counts."""
    keys = ("cut_component_count", "cohesive_component_count", "two_cell_component_count")
    if not all(key in diagnostics for key in keys):
        return ""
    return (f"Компоненты сетки: {diagnostics[keys[0]]:g} · "
            f"с учётом сцепления: {diagnostics[keys[1]]:g} · "
            f"двухъячеечные компоненты сетки: {diagnostics[keys[2]]:g}\n"
            "Число компонентов не является числом устойчивых плит.")
