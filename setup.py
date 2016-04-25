from distutils.core import setup

with open('./gireds/.version', 'r') as verfile:
    __version__ = verfile.read().strip('\n')

packdata = {}
packdata['gireds'] = ['example.cfg', '.version', 'data/*']

setup(name='gireds',
      version='0.1.0',
      packages=['gireds',
                'gireds.utils',
                'gireds.scripts',
                'gireds.pipeline'],
      author='Daniel Ruschel Dutra',
      author_email='druscheld@gmail.com',
      url='https://git.cta.if.ufrgs.br/ruschel/gireds',
      package_data=packdata)
