# DLNLP_LDA
本篇论文使用LDA模型进行文本主题的分类

使用语言：python 3.9

安装以及使用的库：
os,
re,
jieba,
pandas,
numpy,
sklearn,
pyLDAvis

需要注意的是，由于安装的sklearn库的版本比较高，老版的.get_feature_names()需要更改使用.get_feature_names_out()。并且由于sklearn更新较快，而与之适应的pyLDAvis版本仍未更新，运行过程中如果提示出错，可以注释162、164、186、187行再运行。如果想要完成可视化，需要打开pyLDAvis的prepare.py文件，将243行的代码改为default_term_info = default_term_info.sort_values(by='saliency', ascending=False).head(R).drop('saliency', axis=1)

运行前可以自行调整相关参数，来测试不同的情况。需要注意的是，由于只使用单个字进行分类的效果过差，所以默认使用分词进行分类，如果想要尝试使用单个字，请将代码的144行注释，并将145行取消注释；将代码的168行注释，并将169行取消注释即可。
