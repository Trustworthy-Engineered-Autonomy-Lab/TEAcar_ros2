from setuptools import setup

package_name = 'nn_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f'{package_name}.plugins'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='difei',
    maintainer_email='chen.difei@ufl.edu',
    description='Neural network controller with pluggable inferencer',
    license='MIT',
    entry_points={
        'console_scripts': [
            'nn_controller_node = nn_controller.nn_controller_node:main',
        ],
    },
)

