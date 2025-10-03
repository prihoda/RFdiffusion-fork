from setuptools import setup, find_packages

setup(name='rfdiffusion',
      version='1.1.0',
      description='RFdiffusion is an open source method for protein structure generation.',
      author='Rosetta Commons',
      url='https://github.com/RosettaCommons/RFdiffusion',
      entry_points={
          'console_scripts': [
              'rfdiffusion = rfdiffusion.run_inference:main',
          ],
      },
      packages=find_packages(),
      install_requires=[],
      include_package_data=True
)
