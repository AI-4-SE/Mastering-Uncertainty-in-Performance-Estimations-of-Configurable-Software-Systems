import matplotlib
from matplotlib import pyplot as plt
from xml.etree import ElementTree
import datetime
import os
import pickle
import pprint



class SaverHelper:
    def __init__(self, path, dpi=800, fig_pre="fig"):
        self.path = path
        self.session_dir = None
        self.dpi = dpi
        self.fig_pre = fig_pre

    @staticmethod
    def get_clean_file_name(f_name):
        f_name_clean = f_name.replace(" ", "")
        return f_name_clean

    def store_xml(self, xml_root, f_name, folder='.'):
        f_name_clean = self.get_clean_file_name(f_name)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        file_xml = os.path.join(current_folder, 'run-conf-{}.xml'.format(f_name_clean))
        # xml_string = pprint.pformat(xml_root)
        xml_string = ElementTree.tostring(xml_root).decode("utf-8")
        xml_string = '\n'.join([x for x in xml_string.split("\n") if x.strip() != ''])

        with open(file_xml, 'w') as f:
            f.write(xml_string)
        abs_path = os.path.abspath(file_xml)
        return abs_path

    def store_dict(self, r_dict, f_name, folder='.'):
        f_name_clean = self.get_clean_file_name(f_name)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        file_pickle = os.path.join(current_folder, 'results-{}.p'.format(f_name_clean))
        file_txt = os.path.join(current_folder, 'results-{}.txt'.format(f_name_clean))
        with open(file_pickle, 'wb') as f:
            pickle.dump(r_dict, f)
        dict_string = pprint.pformat(r_dict)
        with open(file_txt, 'w') as f:
            f.write(dict_string)
        abs_path = os.path.abspath(file_pickle)
        return abs_path

    def store_pickle(self, obj, f_name, folder='.'):
        f_name_clean = self.get_clean_file_name(f_name)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        file_pickle = os.path.join(current_folder, 'results-{}.p'.format(f_name_clean))
        with open(file_pickle, 'wb') as f:
            pickle.dump(obj, f)
        abs_path = os.path.abspath(file_pickle)
        return abs_path

    def store_figure(self, f_name_clean, folder='.'):
        f_name_clean = self.get_clean_file_name(f_name_clean)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        for extension in ('pdf', 'png'):
            f_name_final = '{}-{}.{}'.format(self.fig_pre, f_name_clean, extension)
            f_path = os.path.join(current_folder, f_name_final)

            plt.savefig(f_path, dpi=self.dpi)

    def set_session_dir(self, name):
        self.session_dir = name
        return self.get_cwd()

    def safe_folder_join(self, *args):
        path = os.path.join(*args)
        os.makedirs(path, exist_ok=True)
        return path

    def get_cwd(self):
        cwd = os.path.join(self.path, self.session_dir)
        return cwd

def get_time_str(i=None):
    i = datetime.datetime.now() if i is None else datetime.datetime.fromtimestamp(i)
    time_str = '-'.join((str(intt) for intt in [i.year, i.month, i.day, i.hour, i.minute, i.second]))
    return time_str
