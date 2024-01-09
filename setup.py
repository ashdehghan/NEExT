import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name='NEExT',
	version='0.2.3',
	author='Ashkan Dehghan',
	author_email='ash.dehghan@gmail.com',
	description='Unsupervised Graph Analysis Framework.',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url='https://github.com/ashdehghan/NEExT',
	license='BSD-2',
	packages=['NEExT'],
	install_requires=[
	"pandas==2.0.3",
	"arrow==1.2.3",
	"tqdm==4.65.0",
	"scikit-learn==1.3.0",
	"matplotlib==3.7.2",
	"scipy==1.11.2",
	"networkx==2.8.8",
	"node2vec==0.4.6",
	"vectorizers==0.2",
	"scipy==1.11.2",
	"numpy==1.25.2",
	"karateclub==1.2.2",
	"umap-learn==0.5.4",
	"jupyter==1.0.0",
	"plotly==5.18.0",
	"xgboost==2.0.2",
	"igraph==0.11.3"]
)