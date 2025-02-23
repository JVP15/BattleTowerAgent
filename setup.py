from setuptools import setup

setup(
    name='BattleTowerAgent',
    version='0.2.0',
    packages=['battle_tower_agent'],
    url='',
    license='',
    author='JVP15',
    author_email='',
    description="An Agent that plays Pokémon Platinum's Battle Tower",
    install_requires=[
        # core libraries needed to run the agent
        'numpy',
        'opencv-python',
        'py-desmume',
        # required to run the DB server
        'flask',
        # required for
        'google-generativeai',
        'google-ai-generativelanguage'
    ]
)
