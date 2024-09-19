def bool_string(value: str) -> bool:
    if value.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no', 'nope', 'never']:
        return False
    else:
        raise ValueError(f"Invalid input value: {value}")