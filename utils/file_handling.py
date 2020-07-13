import warnings
import os


class FileExistsWarning(Warning):

    def __init__(self, *args):
        super(FileExistsWarning, self).__init__(*args)


class NewDirectoryWarning(Warning):

    def __init__(self, *args):
        super(NewDirectoryWarning, self).__init__(*args)


def check_open_name(file_name):
    if os.path.exists(file_name):
        return True
    else:
        warnings.warn("File '%s' does not exist, doing nothing" % file_name, FileExistsWarning)


def check_store_name(file_name, overwrite = False, make_directories = True):
    """
    Checks if the file name already exists, if that is the case it either overwrites or not.
    returns the "new" file name when not overwriting so make sure that you use it as:

    :example
    file_name = os.path.join(os.getcwd(), my_file_name)
    file_name = check_store_name(file_name)

    Moreover if there is an unknown directory it makes the new directory. (It only does this for one layer)

    :param file_name:
    :param overwrite:
    :return:
    """
    # Check if the directory exists if it does do nothing else make the dirs depending or raise error
    # depending on arguments
    if file_name[0]!='/':
        file_name = os.path.join(os.getcwd(), file_name)
        print("APPENDED OS.GETCWD()")

    dir_name = os.path.join(*os.path.split(file_name)[:-1])
    if not os.path.isdir(dir_name):
        if make_directories:
            warnings.warn("Directory did not exist, made new directories: %s" % (os.path.relpath(dir_name)),
                          NewDirectoryWarning)

            os.makedirs(dir_name)
        else:
            open(file_name)

    if os.path.exists(file_name):
        if overwrite:
            return file_name
        else:
            index = 0
            while True:
                split_file_name = file_name.split(".")
                if len(split_file_name)>1:
                    new_file_name = "".join(split_file_name[:-1]) + str(index) + "." + str(split_file_name[-1])
                else:
                    new_file_name = file_name + str(index)
                if not os.path.exists(new_file_name):
                    warnings.warn("File '%s' already existed, appending an %i" % (file_name, index), FileExistsWarning)
                    break
                index += 1
            return new_file_name

    return file_name


