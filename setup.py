from setuptools import setup,find_packages

setup(
    name='Noise_Estimation_of_DEMs',
    version='0.1.0',    
    packages=find_packages(),
    description='Estimating noise of detector error models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/eva-takou/Noise_estimation_of_DEMs',
    author='Evangelia Takou',
    author_email='evangelia.takou@duke.edu',
    license='Apache 2.0',
    #packages=['Noise_estimation_of_DEMs'],
    install_requires=['numpy'],
                      
                      )