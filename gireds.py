#!/usr/bin/env python

import os
import ConfigParser
import time

class pipeline():
    """
    GIREDS - Gmos Ifu REDuction Suite
    
    This class is intended to concentrate and streamline all the of the
    reduction steps for the reduction of Integral Fiedl Unit (IFU) data
    from GMOS.
    """

    def __init__(self, config_file):

        config = ConfigParser.SafeConfigParser()
        config.read(config_file)
        self.cfg = config

        # Define directory structure
        self.root_dir = config.get('main', 'root_dir')
        self.raw_dir = config.get('main', 'raw_dir')
        self.products_dir = config.get('main', 'products_dir')
        self.run_dir = self.products_dir + time.strftime('%Y-%M-%dT%H:%M:%S')

        try:
            os.mkdir(self.products_dir)
        except OSError as err:
            if err.errno == 17:
                pass
            else:
                raise e

        os.mkdir(self.run_dir)


    def associate_files(self):
       
        pass        



if __name__ == "__main__":
    import sys
    pip = pipeline(sys.argv[1])
    #pip.associate_files()
