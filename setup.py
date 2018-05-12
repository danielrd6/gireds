import shlex
from subprocess import check_output
from distutils.core import setup

git_head_rev = check_output(
    shlex.split('git rev-parse --short HEAD')).strip()

with open('./gireds/.version', 'r') as verfile:
    __version__ = verfile.read().strip('\n')

packdata = {}
packdata['gireds'] = ['example.cfg', '.version', 'data/*']

setup(name='gireds',
      version=__version__,
      packages=['gireds',
                'gireds.utils',
                'gireds.pipeline'],
      scripts=['bin/gireds',
               'bin/gemini_archive',
               'bin/make_bpm',
               'bin/auto_apertures',
              ],
      author='Daniel Ruschel Dutra',
      author_email='druscheld@gmail.com',
      url='https://git.cta.if.ufrgs.br/ruschel/gireds',
      package_data=packdata,
      options=dict(egg_info=dict(tag_build='dev_' + git_head_rev)),
      platform='Linux',
      license='GPLv3',
      description='Program to reduce GMOS IFU data.')
