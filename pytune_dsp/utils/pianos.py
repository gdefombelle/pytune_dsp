def infer_era_from_year(year: int) -> str:
    """
    Infers the piano era from the year of manufacture.
    """
    if year < 1880:
        return "antique"
    elif 1880 <= year < 1970:
        return "vintage"
    else:
        return "modern"

def probability_strings_per_note(midi: int, piano_type: str = "upright", era: str = "modern") -> dict[int, float]:
    """
    Retourne une probabilité du nombre de cordes (1,2,3) pour une note donnée,
    en tenant compte du type de piano et de son époque.
    """
    # Base: configuration générique
    if midi < 32:
        base = {1: 0.9, 2: 0.1, 3: 0.0}
    elif midi < 45:
        base = {1: 0.2, 2: 0.7, 3: 0.1}
    elif midi < 61:
        base = {1: 0.0, 2: 0.5, 3: 0.5}
    else:
        base = {1: 0.0, 2: 0.1, 3: 0.9}

    # Ajustements selon type
    if piano_type == "upright":
        # pianos droits → transitions plus précoces, plus de 2 cordes
        if midi < 32:
            base = {1: 0.95, 2: 0.05, 3: 0.0}
        elif midi < 45:
            base = {1: 0.3, 2: 0.6, 3: 0.1}
        elif midi < 61:
            base = {1: 0.0, 2: 0.6, 3: 0.4}
    elif piano_type == "grand":
        # pianos à queue → médium avec plus de 3 cordes
        if midi < 45:
            base = {1: 0.1, 2: 0.8, 3: 0.1}
        elif midi < 61:
            base = {1: 0.0, 2: 0.3, 3: 0.7}

    # Ajustements selon l’ère
    if era == "antique":
        # avant 1880 : moins de 3 cordes
        base[3] *= 0.5
        base[2] += 0.25
        base[1] += 0.25
    elif era == "vintage":
        # 1900–1950 : transition vers moderne
        base[3] *= 0.8
        base[2] += 0.2
    elif era == "modern":
        # >1970 : standard actuel
        pass

    # Normalisation
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}