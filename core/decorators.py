from deprecated import deprecated

def deprecated(func):
    """Décorateur neutre temporaire pour marquer une fonction comme obsolète (sans effet)."""
    return func
