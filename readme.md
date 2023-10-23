(没时间写了，找时间再补充吧)

#  Intelligent complementary diagnosis of degenerative OA



###    一、本项目实施半年内取得的主要成果，达到的目标、水平，创新之处  

​    我们的项目旨在遵循中医思维，开发智能辅助诊断工具，用于退化性骨关节病的早期检测和诊断。我们的研究分为数据准备、模型构建、结果分析和平台搭建四个部分，并已经在各个环节取得了重要的进展。     

**1.** **数据集介绍**  在数据准备的部分，我们着重对数据进行了收集、预处理以及特征选择。以下是我们在这一阶段所取得的进展和具体措施：     

**1.1 一般资料**  (1) 研究患者来源： 本研究的患者均为2011-2021年间就诊于暨南大学附属第一医院、佛山市中医院、五邑中医院骨关节科、湖南中医药大学附属第一医院的门诊和住院患者。  (2) 样本数量： 我们总共纳入了5013例样本，其中包括健康样本601例和膝骨关节炎患病样本4412例。  (3) 数据集变量： 数据集包含了556个变量，其中10个变量用作标记，涉及膝骨关节炎、平和质、气虚质、阳虚质、阴虚质、痰湿质、湿热质、血瘀质、气郁质以及特禀质等。     

**1.2 诊断标准**  我们使用了《膝骨关节炎中医诊疗指南(2020年版)》以及Kellgren-Lawrence影像分级作为膝骨关节炎的诊断标准。     

**1.3 纳入标准**  (1) 男女性别不限。  (2) 所有患者签署了知情同意书。     

**1.4 数据预处理和特征选择**  我们通过对病人的原始信息进行批量汇总，并对信息进行脱敏处理，对异构数据进行合并，整理成为表格型数据。在此之上对原始数据进行了详细的清洗，为了降低数据的稀疏性，控制正负样本比例，加快模型收敛速度，对数据集进行降采样、空缺值填补、特征分解、对频数为零的特征进行剔除以及归一化处理。最后得到 1421 例样本作为训练集，其中  KOA 患者样本 1111 例，男性患者 309 例，女性患者 802 例，平均年龄为 66.40 岁，平均体重 59.51 千克。1.5 数据说明数据集一共约540个特征，字段的含义因医疗数据保密原因无从得知。  对于数据集中的变量，除了性别外，我们采用二进制编码方式，其中0表示“否”，1表示“是”，以方便后续的数据分析和模型建立，使得数据在计算机环境下更易于处理。例如，以“ggjbwh1”为例，它表示“是否正常”，其中0代表“不正常”，1代表“正常”。同样，对于其他带有数字后缀的变量，数字反映了具体的属性，如“ggjbwh2”表示“是否自汗”，“ggjbwh4”表示“是否出现冷汗”，以此类推。     ![img](https://s2.loli.net/2023/10/23/9z7rOfTjXWbhESi.jpg)  图 1 数据集预览表     

  上图所示为数据集的预览表，可以看出数据集特征众多，因此我们在进行深度学习实验之前必须首先进行特征预处理，包括数据清洗、缺失值处理、异常值检测和规范化等步骤，以确保数据的质量和一致性。

**2.** **数据预处理**

**2.1** **数据分析**

在数据的预处理过程中，我们采用了皮尔逊相关性分析方法，以探索性状之间的相关性和相互影响。如下图所示，分析结果显示，每个特征相对于其他特征在一定程度上表现出独立性。这意味着每个性状或特征在某种程度上不依赖于其他特征，具有一定的独立性。尽管特征相对独立，但我们也观察到它们可以共同对预测结果和性状产生影响。这表明特征之间存在某种程度的相互作用，它们可能一起对患者的健康状况和预测结果产生影响。以下两张图展示了所有特征之间和仅性状之间的相关性热图。

 

| ![pearson_pca_TCM](https://s2.loli.net/2023/10/23/UDn3QCoaEPJRYbc.png)图 2 所有特征之间相关性图 2 所有特征之间相关性 | ![pearson_heart](https://s2.loli.net/2023/10/23/fvXQKzEPTk5MWAg.png)图 3 仅性状的相关性 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

 

![label_corrplot](https://s2.loli.net/2023/10/23/jBlyWHGhSAk6DrK.jpg)

图 4 进行性状相关性排序

如图，性状之间具有相关性，存在着相互影响，相互干预的状态。因此，我们选择将OA（骨关节炎）的预测问题看作一个多标签分类任务，其中每个性状或特征可以被视为一个可能的标签，目标是使用这些性状特征来预测患者是否患有OA，以及可能的其他相关疾病或症状，这有助于更全面地捕捉患者的健康状态，并提高模型对于疾病预测的准确性。

 

**2.2** **特征工程**

特征相关性排序：我们使用随机森林算法对特征进行相关性分析，并将特征按其对疾病状态的相关性进行排序（如表1所示），确定了对于模型性能的贡献较大的特征。特征相关性阈值：在完成相关性分析后，我们设定了一个相关性阈值，用于确定哪些特征的相关性低于一定程度，可能对疾病状态的预测贡献较小。PCA降维处理：针对那些与疾病状态相关性低于阈值的特征，我们采用主成分分析（PCA）方法来进行降维处理。PCA旨在保留数据中的主要变化，同时减少维度。这有助于简化模型并减少不必要的特征，从而提高模型的计算效率和泛化能力。

 

 

表 1 相关性排序表

|                  | Correlation_y0 | abs      |
| ---------------- | -------------- | -------- |
| commonSsHs       | 1.409194       | 1.409194 |
| commonXbYs2      | 1.379987       | 1.379987 |
| ggjbZzGjhdsmcy   | -1.320017      | 1.320017 |
| ggjbJzhjMt       | -1.240481      | 1.240481 |
| commonTzDpt4     | 1.171614       | 1.171614 |
| commonTzYpf2     | 1.168342       | 1.168342 |
| commonSxM1ys3    | -1.128112      | 1.128112 |
| commonTzNsyhjbh2 | -1.123149      | 1.123149 |
| commonDbYs3      | -1.110721      | 1.110721 |
| commonMzChe      | 1.083196       | 1.083196 |
| commonSsQzs      | -1.042144      | 1.042144 |
| commonTzJsjz4    | -1.017144      | 1.017144 |
| commonTzMsha1    | 0.997876       | 0.997876 |
| commonTzFbfm2    | -0.944944      | 0.944944 |
| commonSg         | -0.925406      | 0.925406 |
| ggjbTtxzBtjz     | -0.923651      | 0.923651 |
| ...              | ...            | ...      |
| commonTzKc1      | 0.000000       | 0.000000 |

 

 

**3.** **训练方法**

在此次研究中，我们采用了多种学习方法，包括机器学习经典算法如决策树、高斯朴素贝叶斯、K最近邻、逻辑回归、随机森林、随机梯度下降等。

在深度学习方面，我们采用了多层神经网络进行模型训练，以更好地学习数据中的复杂特征表示，提高模型性能。此外，为了减少参数量和提高计算效率，我们引入了主成分分析法（PCA）来进行特征降维，并采用了自适应学习率方法以优化训练过程。

 

**3.1** **机器学习**

在本次研究中，模型构建部分我们首先在公开数据集上使用各种机器学习模型进行预训练，并进行效果比对，包括决策树、Gaussian、KNN、LinearSVM、Logreg、Perceptron、Random Forest、SGD、SVM。

![models_results_heart](https://s2.loli.net/2023/10/23/3KG5FoRrqOiSvHc.png)

图 5 模型比较图

 

1.1 深度学习

多标记分类方法是机器学习中一个重要研究方向，被广泛使用于多语义分割问题。例如在文本分类中，每个文本都可能属于多个主题。在对这样的文本进行分类时，每一个主题可以看作是一个标签，如果用一个标签代表一篇文章的主题，可能过于笼统，多标记学习框架则专门用于解决这类问题。在中医临床实际情况中，简单证型较少出现，一个病例往往有多个症状，可能出现多种相关证候，多个证型兼夹的情况较为常见。

 

从信息学角度来看，多证型兼夹的中医诊断过程属于典型的多标记学习问题。根据算法对标记相互关系利用的策略，多标记学习又分为低阶策略和高阶策略。例如一个患者可能同时存在气虚、气滞血瘀和心脾不交等多个中医证型，低阶策略下的多标记学习往往把这种多证兼夹的情况分为单个证型或者两两组合进行处理，这样的方式忽略了其他证型对目标证型的影响。而高阶策略的多标记算法则可以避免忽略标签相关性这个问题，更符合中医临床出现的多症状兼夹情况。

 

基于此，在训练和测试过程中我们构建了深度学习模型。深度学习模型能够更好地捕捉这些复杂的多标签关系，从而更好地适应中医数据集中的多标签分类问题，有助于提高模型的性能，以获得更高的准确率，使其更适用于中医领域的数据分析和诊断任务。

 

1.2 迁移学习

我们在公开数据集上预训练了多种常用分类模型进行效果比对，如图7所示，并将预训练模型（图6）的参数迁移到中医数据集上，发现准确性比较中，随机森林为58.4507，为四个数值中最大值，效果最好。如图7所示。

 

|      |                                                              |      |                                                              |
| ---- | ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
|      | ![pre_models_acc](https://s2.loli.net/2023/10/23/TIpWoMv3Bs6mLxb.png) |      | ![model_transfer](https://s2.loli.net/2023/10/23/cd376gbn2J1HfaI.png) |
|      | 图 6 预训练模型                                              |      | 图 7 迁移模型                                                |

**3.4 主成分分析法PCA**

主成分分析法（PCA）被广泛用于特征降维，以减少数据集中的冗余信息和降低模型的复杂性。在我们的项目中，特征的数量达到数百个，相关性较低的变量对训练过程具有冗余性，所以研究并使用PCA降维能加快训练过程的同时保持原有的准确度，并且防止模型受无关因素干扰。

 

PCA通过寻找数据中的主成分（特征）来实现降维，从而保留了最重要的信息。其核心思想是通过线性变换，将原始特征映射到新的坐标系，其中数据的方差最大化。

 

其计算为：

(1) 计算特征均值

 

(2) 中心化数据：每个特征减去均值

 

meanVals = np.mean(dataMat, axis=0)

(3) 计算协方差矩阵

covMat = np.cov(meanRemoved, rowvar=0)

 

(4) 计算协方差矩阵的特征值和特征向量

covMat = np.cov(meanRemoved, rowvar=0)

 

(5) 对特征向量按特征值大小排序

eigValInd = np.argsort(eigVals)

 

(6) 选择前N个特征向量作为主成分

redEigVects = eigVects[:, eigValInd]

 

(7) 将原始数据投影到主成分空间

reduced_data = np.dot(centered_data, top_eigenvectors)

**3.5 自适应学习率**

在我们的项目中，自适应学习率方法用于优化深度学习模型的训练过程，以提高模型的收敛速度和性能。

自适应学习率的核心思想是根据模型训练的进展情况动态地调整学习率，以确保在训练的早期使用较大的学习率以加速收敛，而在后期逐渐减小学习率以稳定模型训练。

其中，我们采用了AdaGrad（自适应梯度算法）作为自适应学习率方法的代表，其公式为：learning_rate = initial_learning_rate / sqrt(sum_of_squared_gradients + epsilon)

其计算如下：

(1) 初始化初始学习率 (initial_learning_rate)

(2) 初始化梯度平方和 (sum_of_squared_gradients)为0

(3) 在每次迭代中计算梯度

(4) 更新梯度平方和：sum_of_squared_gradients += gradient^2

(5) 计算自适应学习率：

learning_rate = initial_learning_rate / sqrt(sum_of_squared_gradients + epsilon)

(6) 使用自适应学习率进行参数更新



**4.** **训练结果**

 

为了提高分类准确度，我们在中医数据集上使用了深度学习方法并建立模型，我们首先使用深度学习方法，花费了大量的时间训练模型，该模型考虑了所有可用的特征。虽然这种方法能够在某种程度上取得令人满意的结果，但训练时间较长，绘制的准确率曲线如图8所示；

 

![Accuracy1](https://s2.loli.net/2023/10/23/lzvCJhf1xrBMsp7.png)

图 8 Accuracy1

 

 

 

为了减少模型的参数数量，我们采用主成分分析法（PCA），保留了数据中的主要信息，并减少冗余特征，绘制的准确率曲线如图9所示，我们发现这种方法并没有使模型充分收敛，因此效果有限；

 

![Accuracy2](https://s2.loli.net/2023/10/23/q5wMx1WZ9K3hrYy.png)

图 9 Accuracy2

 

最后，我们选择了自适应学习率方法，以保证学习的效率和模型参数的最佳化。自适应学习率方法可以根据模型的收敛情况来动态地调整学习率，以确保模型在训练过程中能够更好地收敛到最优解，这一步的结果如图10所示，可以看出，相同时间内，原模型只能训练到50epoch，而经过自适应学习率操作后的模型能够更快地收敛，但性能仍不够理想；

 

![bd0c2a219663075060c7551ba20739d](https://s2.loli.net/2023/10/23/SLHdeqWnMPaTtNF.jpg)

图 10 Accuracy3

 

为了进一步改进模型性能，我们根据特征相关性扩大了PCA特征选择的范围，选择了大约200个特征。这一步取得了更快速达到高准确率的结果。

 

 

![0e3e24778053eb55e0914f159924242](https://s2.loli.net/2023/10/23/cHa4TEJytM9kqIR.jpg)

图 11 Accuracy4



图12所示的内容为四种处理方法前后的损失曲线图，也可以看出经过自适应学习率方法与PCA降维处理过后的模型损失率更小，性能更好。

 

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![f058e31374459b485114d2ef1fb8f19](https://s2.loli.net/2023/10/23/X9aq6O4AeI7k3hY.jpg) |


图 12 loss



 

 

除了准确率（Accuracy）和损失（Loss）作为评估模型性能的指标外，我们还采用了其他指标来综合考虑特征数量的影响和PCA降维的有效性，包括：精确度（Precision）： 衡量了模型在预测阳性类别时的准确性，有助于了解模型的误报率。召回率（Recall）： 衡量了模型能够识别真正阳性样本的能力，有助于了解模型的漏报率。F1分数（F1-Score）： 结合了精确度和召回率，提供了对模型整体性能的综合评估。Support：用来衡量在每个类别中的样本数量，能够更好地了解PCA降维对不同类别的样本数量分布的影响，以及模型在不同类别中的表现。通过综合考虑这些评估指标有助于我们更好地选择适当的特征数量，以提高多标签分类任务的效果。

 

![img](https://s2.loli.net/2023/10/23/cHa4TEJytM9kqIR.jpg)

图 13 其他指标图



**5.** **平台搭建**

在平台搭建阶段，我们已经完成了需求分析、原型图设计、系统架构与环境搭建等工作。我们正着手进行前后端开发，以将我们的研究成果转化为可用的小程序和网站。我们计划在项目结束时成功部署上线，并持续进行测试和维护工作。

总的来说，我们的项目已经取得了阶段性的进展。我们运用深度学习的方法，通过与多种机器学习算法的较对以及PCA降维，自适应学习率的方法结合为退化性骨关节病的诊断提供了新的解决方案。

虽然我们已经实现了具有很高准确性的预测效果，但我们认识到仅仅覆盖一小部分患者是不够的。我们的下一步目标是扩大人群覆盖范围，以使更多人受益于我们的智能辅助诊断工具。这将涉及更广泛的数据收集和更多的合作伙伴。

除此之外，我们计划在项目的后续阶段着手搭建一个用户友好的平台，以帮助公众更轻松地进行预测。这将包括需求分析、原型图设计、系统架构与环境搭建、前后端开发、测试等多个环节，以确保平台的高效运行和用户满意度。我们在这个项目中取得的进展是基于中医思维的先进技术和深刻的数据分析的有机结合。

**2.** **项目特色与创新点**

人工智能技术与智能辅助诊断结合，可以降低中医诊断领域的误诊率，提高中医诊断的准确度。将传统中医“望闻问切”的方法与先进的特征选择、深度学习方法结合，是推动中医现代化发展的重大举措，也是该项目的主要特色和创新点。开展面向中医行业的人工智能模型和算法的研发，推进在退化性骨关节病的早期诊断场景形成应用创新和应用方案，是国家重点支持领域项目申报的范畴，也是本项目团队致力研究的目标。

在理论方面，该项目遵循了中医思维规律，关注患者健康状况的各种风险性因素，以个性化诊断与评估为核心，深度融合人工智能、云计算、大数据技术等技术。通过系统整理中医的科学知识和经验知识，建立符合计算机语言的数据结构及知识规则，构建一个基于深度学习和特征工程的网络模型。同时运用大数据、云计算的理论，搭建退化性骨关节病的中医智能辅助早期诊断平台。依赖互联网实现中医远程诊疗，从而有效弥补中医技术短板，充分发挥中医简便廉验的特点。

在技术方面，该项目结合大数据、云计算和人工智能技术，采用特征工程、深度学习的原理，实现从原数据采集降维，到模型构建和完善、再到成果转换和部署的一系列过程通过这些技术手段，实现中医诊疗客观化、智能化，为中医治疗退化性骨关节病的理论和临床研究提供技术支撑。这将提高中医诊疗的准确性和治疗效果，为患者提供更好的医疗服务。

 

### 二、此项目在研期间遇到的困难及解决困难的思路

**1、** **1.** **
** **数据集获取问题**

(1) 目标：在符合相关法律法规的条件下，使数据集尽可能的完善。

(2) 困难：由于医院的保密制度和和相关数据涉及到病人隐私，虽然数据集中所包含的数据从整体上看较为齐全，但是仍然存在着部分特征大量缺失的情况。如何在符合个人隐私保护法律和医疗数据保护法规的条件下，更好的实现与相关机构或个人进行数据共享，是目前项目推进的难点之一。

(3) 思路：了解相关法律法规，包括个人隐私保护法律和医疗数据保护法规，确保在获取和使用病患数据时遵守相关规定。建立与医院和医疗机构的合作关系。

与数据提供方建立合作关系，例如与公司、研究机构或政府合作，以获取他们的数据集。通过签署合作协议或数据共享协议来确保数据的安全和隐私。在合作过程中，开发数据采集和整合系统，自动化地从医院信息系统中提取病患数据。这样可以减少人工操作和错误，并提高数据获取的效率和准确性。

普及公众教育和获取知情同意，向公众普及数据共享的重要性和好处，并征得病患的知情同意。通过教育和透明的沟通，增加公众对数据共享的理解和支持。

**2、** **2.数据安全问题**

(1) 目标：利用加密技术和脱敏技术使数据在使用过程中更加安全。

(2) 困难：在处理和分析数据时，隐私问题是一个重要的考虑因素，因此数据安全问题成为了项目一大难点。数据隐私问题涉及到如何保护个人身份和敏感信息，以及如何确保数据在处理和传输过程中不被未经授权的人访问或泄露。

(3) 思路：在获取病患数据时，对数据进行脱敏和匿名化处理，如替换、删除或模糊化等，以保护病患的隐私和个人信息安全。确保在使用数据时无法识别个人身份。在传输和存储过程中采取必要的安全措施，包括加密、访问控制、备份和灾难恢复等。通过定期对数据进行备份，防止因硬件故障、病毒攻击等原因导致的数据丢失，并通过定期进行安全审计，检查系统和数据的安全性，发现并修复潜在的安全漏洞。利用此类方法，以确保数据的安全性，防止数据的泄露和滥用。

**3、** **3. 多标签分类所带来的相关问题**

(1) 目标：采用多标签分类算法和模型集成，使得分类结果更准确。

(2) 困难：相比于普通的分类任务，该模型中所需要使用到的多标签分类任务更加困难：

样本的标签空间更大，增加了分类的复杂性；多标签分类任务中，不同标签之间可能存在相关，使得模型需要能够捕捉到标签之间的相关性以更准确地进行分类；不同标签的出现频率可能不平衡，使得模型需要能够处理样本不平衡问题，以避免对常见标签的过度关注而忽略罕见标签。

(3) 思路：使用专门针对多标签分类任务的算法，如基于二分类的方法、基于决策树的方法等。并对输入数据进行特征提取和处理，以提高分类模型的性能和准确度。除此之外，还可以通过使用多个分类模型进行集成，如投票、平均、堆叠等方法，以提高模型的分类效果和泛化能力。而在处理多标签分类任务中可能存在的类别不平衡问题方面，可以使用过采样、欠采样、SMOTE等方法。
