import os
def count_folders_os(path):
    """
    Count folders using os.scandir() - fastest method for large directories
    Returns only immediate subdirectories (not recursive)
    """
    try:
        # Count only directories in the immediate path
        return sum(1 for entry in os.scandir(path) if entry.is_dir())
    except PermissionError:
        print(f"Permission denied to access {path}")
        return 0
    except FileNotFoundError:
        print(f"Path not found: {path}")
        return 0