from setuptools import setup

setup(
    name='diffraction',
    version='0.1',
    url='http://github.com/noahwaterfieldprice/diffraction/',
    author='Noah Waterfield Price',
    author_email='noah.waterfieldprice@physics.ox.ac.uk',
    description='A package for simulating diffraction experiments'
                'and performing crystallographic calculations.',
    packages=['diffraction', 'diffraction.cif']
)
