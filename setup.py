from setuptools import setup, find_packages

setup(
    name='project-name',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of the project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/project-name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
        # e.g., 'numpy', 'pandas', 'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)