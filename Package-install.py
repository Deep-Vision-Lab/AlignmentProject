import pkg_resources

def save_packages_to_txt(output_file='requirements.txt'):
    """
    Save all installed Python packages and their versions to a text file (requirements.txt).
    This script lists only the package names and their versions (no local paths).
    """
    try:
        # Get the list of installed packages and their versions
        packages = pkg_resources.working_set

        # Open the output file
        with open(output_file, 'w') as f:
            # Write each package and its version to the file
            for dist in packages:
                f.write(f"{dist.project_name}=={dist.version}\n")

        print(f"Packages and their versions have been written to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
if __name__ == '__main__':
    save_packages_to_txt('requirements.txt')
