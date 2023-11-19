""" Class to manage the Folders/Subfolders Paths for wxperiments"""
import ntpath
import pathlib
import numpy
from .utils_general import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PIPE = "|"
ELBOW = "|__"
TEE = "|--"
PIPE_PREFIX =  "|   "
SPACE_PREFIX = "    "


# Directory Tree Printer / Organizer
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def files_in_folder(folder, extension='dicom', sort=False):
    """
    This function returns the list of files in a folder, with a specific extension and sorted.
    Args:
        folder: Path to the folder
        extension: extension of the files to be selected
        sort: sort the files in alphabetical order

    Returns:
        list of files in the folder
    """

    if extension is None:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    else:
        files = [os.path.join(folder,f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(extension)]
    if sort:
        files.sort()
    return files





class DirectoryTree:
    def __init__(self, root_dir):
        self._generator = _TreeGenerator(root_dir)

    def generate(self):
        tree = self._generator.build_tree()
        for entry in tree:
            print(entry.encode('utf-8'))
        return tree

class _TreeGenerator:
    """
    <_TreeGenerator> Object Class is needed to generete the tree structure of the directory structure.
    FOr the pourpose of the tre construction there are some fixed attributeds
    Project added from https://realpython.com/directory-tree-generator-python/
    """

    def __init__(self, root_dir):
        """
        Class <__init__> constructor. In this case, .__init__() takes root_dir as an argument. It holds the tree’s root directory path.
         Note that you turn root_dir into a pathlib.Path object and assign it to the nonpublic instance attribute ._root_dir.
        :param root_dir: <pathlib.Path> Object
        """
        self._root_dir = pathlib.Path(root_dir)
        self._tree = [] # empty tree of directories

    def build_tree(self):
        """
        Tree building function.
        This public method generates and returns the directory tree diagram. Inside .build_tree(), you first call ._tree_head() to build the tree head.
        Then you call ._tree_body() with ._root_dir as an argument to generate the rest of the diagram.
        :return:
        list, the tree of directories
        """
        self._tree_head()
        self._tree_body(self._root_dir)
        return self._tree

    def _tree_head(self):
        """
        This method adds the name of the root directory to ._tree. Then you add a PIPE to connect the root directory to the rest of the tree.
        :return: None
        """
        self._tree.append(f"{self._root_dir}{os.sep}")
        self._tree.append(PIPE)

    def _tree_body(self, directory, prefix=""):
        """

        :param directory:  holds the path to the directory you want to walk through. Note that directory should be a pathlib.Path object.
        :param prefix: holds a prefix string that you use to draw the tree diagram on the terminal window.
        This string helps to show up the position of the directory or file in the file system.
        :return: None
        """
        entries = directory.iterdir()
        entries = sorted(entries, key=lambda entry: entry.is_file()) # This function iterate the generator of paths <entries>
        entries_count = len(entries)
        for index, entry in enumerate(entries):
            connector = ELBOW if index == entries_count - 1 else TEE
            if entry.is_dir():
                self._add_directory(
                    entry, index, entries_count, prefix, connector
                )
            else:
                self._add_file(entry, prefix, connector)

    def _add_directory(
            self, directory, index, entries_count, prefix, connector
    ):
        self._tree.append(f"{prefix}{connector} {directory.name}{os.sep}")
        if index != entries_count - 1:
            prefix += PIPE_PREFIX
        else:
            prefix += SPACE_PREFIX
        self._tree_body(
            directory=directory,
            prefix=prefix,
        )
        self._tree.append(prefix.rstrip())

    def _add_file(self, file, prefix, connector):
        self._tree.append(f"{prefix}{connector} {file.name}")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def find_last_exp(path_log_run:str, MODE):
    if MODE == 'last':
        import re
        # last path is equal to the path_log
        LIST_EXPS = [[int(re.search(r'\d+', sub_folder.replace('--',':').split(':')[0]).group()),sub_folder] for sub_folder in os.listdir(path_log_run)]
        path_ext = LIST_EXPS[numpy.argmax([float(exp[0]) for exp in LIST_EXPS])][1]
        return os.path.join(path_log_run, path_ext)


def generate_id_for_multi_exps(run_dir_root: str):
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    import re
    ids = [d.split('--')[0]for d in os.listdir(run_dir_root)]
    r = re.compile("^\\d+")
    run_id=1
    for id in ids:
        m = r.match(id)
        if m is not None:
            i = int(m.group())
            run_id = max(1, i + 1)
    return run_id

def get_next_run_id_local(run_dir_root: str, module_name: str) -> int:
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    import re
    #dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    #dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d)) and d.split('--')[1] == module_name]
    dir_names = []
    mkdir(run_dir_root)
    for d in os.listdir(run_dir_root):
        if not 'configuration.yaml' in d and not 'log.txt' in d and not 'src' in d:
            try:
                if os.path.isdir(os.path.join(run_dir_root, d)) and d.split('--')[1] == module_name:
                    dir_names.append(d)
            except IndexError:
                if os.path.isdir(os.path.join(run_dir_root, d)):
                    dir_names.append(d)

    r = re.compile("^\\d+")  # match one or more digits at the start of the string
    run_id = 1

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id
def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]
def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()
    return folders