# Fundamentals


##  Contents
1. [ Semantic Segmentation](#anabaslik1)
    1. [ What is Machine Learning?](#paragraf1)
    2. [What is Unsupervised vs Supervised learning difference? ](#paragraph2)
        1. [Supervised learning](#subparagraph1)
        2. [Unsupervised learning](#subparagraph2)
    3. [What is Deep Learning?](#paragraph3)
   4. [What is Neural Network (NN)?](#paragraph4)
   5. [What is Convolution Neural Network (CNN)?](#paragraph5)
   6. [What is segmentation task in NN?](#paragraph6)
       1. [Semantic segmentation](#subparagraph3)
       2. [Instance segmentation](#subparagraph4)
    7. [What is classification task in NN?](#paragraph7)
    8. [Compare segmentation and classification in NN](#paragraph8)
    9. [ What is data and dataset difference?](#paragraph9)
    10. [ What is the difference between supervised and unsupervised learning in terms of dataset?](#paragraph10)
    11. [ Data Preprocessing](#paragraph11)
          1. [ Extracting Masks](#paragraph12)
              1. [What is color space ?](#paragraph13)
              2. [What RGB stands for ?](#paragraph14)
              3. [ In Python, can we transform from one color space to another?](#paragraph15)
              4. [What is the popular library for image processing?](#paragraph16)
        2. [Converting into Tensor](#paragraph17)
           1. [What is Computational Graph?](#paragraph18)
           2. [What is Tensor?](#paragraph19)
           3. [What is one hot encoding?](#paragraph20)
           4. [What is CUDA programming?](#paragraph21)
       12. [Design Segmentation Model](#paragraph22)
             1. [What is the difference between CNN and Fully CNN (FCNN) ?](#paragraph23)
             2. [What are the different layers on CNN ?](#paragraph24)
                1. [Convolutional Layer](#paragraph25)
                2. [Pooling Layer](#paragraph26)
                3. [Fully Connected Layer](#paragraph27)
           3.  [What is activation function ?](#paragraph28)
       13. [Train](#paragraph29)
            1. [What is parameter and hyper-parameter in NN ?](#paragraph30)
            2. [Validation Dataset](#paragraph31)
            3.  [What is an epoch?](#paragraph32)
            4. [What is batch?](#paragraph33)
            5. [What is iteration?](#paragraph34)
            6. [What Is the Cost Function?](#paragraph35)
            7. [What is/are the purpose(s) of an optimizer in NN?](#paragraph36)
            8. [What is Batch Gradient Descent & Stochastic Gradient Descent?](#paragraph37)
            9. [What is Backpropogation ? What is used for ?](#paragraph38)
2. [ Object Recognition](#anabaslik2)
    1. [What is an image classification task?](#paragraph39)
    2.  [What is an object localization task?](#paragraph40)
    3. [What is an object detection task?](#paragraph41)
    4. [What is an object recognition task?](#paragraph42)
    5.  [What is bounding box regression?](#paragraph43)
    6. [What is bounding box regression?](#paragraph44)
    7.  [What is non-max suppression?](#paragraph45)
   
         
              
                   
        
         
         
           

# Semantic Segmentation<a name="anabaslik1"></a>

## What is Machine Learning?<a name="paragraph1"></a>


With machine learning, humans input data as well as the answers expected from the data, and out come the rules. These rules can then be applied to new data to produce original answers. A machine-learning system is trained rather than explicitly programmed. It’s presented with many examples relevant to a task. It finds statistical structure in these examples that eventually allows the system to come up with rules for automating the task.

(Makine Öğrenmesi, veri ve beklenen çıktıları kullanarak yeni verilere özgün cevaplar üretecek kuralları öğrenir. Makine öğrenmesi sistemleri elle programlanmak yerine **eğitilirler**. Göreve yönelik verilen örneklere bakarak istatistiksel örüntüleri ortaya çıkartıp görevi otomatikleştirecek kuralları ortaya koyar.) 

## What is Unsupervised vs Supervised learning difference?<a name="paragraph2"></a>

### Supervised learning;<a name="subparagraph1"></a>


It consists of learning to map input data to known targets (also called annotations), given a set of examples (often annotated by humans).

(Bir veri seti üzerindeki girdilerden bilinen çıktılara/hedeflere olan eşleştirmeyi öğrenmeyi kapsar.)

### Unsupervised learning;<a name="subparagraph2"></a>

This branch of machine learning consists of finding interesting transformations of the input data without the help of any targets, for the purposes of data visualization, data compression, or data denoising, or to better understand the correlations present in the data at hand. The main distinction between the two approaches is the use of labeled datasets. To put it simply, supervised learning uses labeled input and output data, while an unsupervised learning algorithm does not.

( Makine öğrenmesinin bu kategorisinin amacı veriyi görselleştirmek, verileri sıkıştırmak, verilerdeki gürültüyü azaltmak ya da eldeki veriler arasındaki kolerasyonu anlamak için girdilerden herhangi bir bilinen hedefin yardımı olmaksızın amaca uygun dönüşümleri bulmaktır.) (İki yaklaşım arasındaki temel ayrım, etiketli veri kümelerinin kullanılmasıdır. Basitçe söylemek gerekirse, denetimli öğrenme etiketli girdi ve çıktı verilerini kullanırken denetimsiz öğrenme algoritması kullanmaz.) 

## What is Deep Learning?<a name="paragraph3"></a>

Deep learning is a specific subfield of machine learning: a new take on learning representations from data that puts an emphasis on learning successive layers of increasingly meaningful representations. The deep in **deep learning** isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. How many layers contribute to a model of the data is called the *depth* of the model.

(Derin öğrenme, birbirini takip eden katmanlarda veriler işlenirken giderek artan şekilde daha kullanışlı gösterimler elde edilebilen makine öğrenmesinin bir alt alanıdır. Derin derken kastedilen **derin öğrenmenin** birtakım derin bilgiler elde etmesi değil, birbirini takip eden gösterim katmanları ifade edilmektedir. Modeldeki katman sayısı modelin ***derinliğini*** oluşturmaktadır.) 

## What is Neural Network (NN)?<a name="paragraph4"></a>

Modern deeplearning often involves tens or even hundreds of successive layers of representations and they’re all learned automatically from exposure to training data. Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of representations of the data; hence, they’re sometimes called shallow learning. In deep learning, these layered representations are (almost always) learned via models called neural networks, structured in literal layers stacked on top of each other.

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

(Modern derin öğrenme modelleri, onlarca hatta yüzlerce birbirini takip eden katmanlar içermektedir. Oysa diğer makine öğrenme algoritmaları, genelde bir veya iki katmandan oluşur ki bazen *sığ öğrenme* olarak da adlandırılırlar. Bu katmanlı gösterim, **sinir ağı** diye adlandırılan ve birbirini takip eden katmanları olan model sayesinde öğrenilmektedir.

Yapay sinir ağları (YSA), bir girdi katmanı, bir veya daha fazla gizli katman ve bir çıktı katmanı içeren bir düğüm katmanından oluşur. Her düğüm veya yapay nöron diğerine bağlanır ve ilişkili bir ağırlık ve eşiğe sahiptir. Herhangi bir düğümün çıktısı belirtilen eşik değerinin üzerindeyse, o düğüm etkinleştirilir ve ağın bir sonraki katmanına veri gönderilir. Aksi takdirde, ağın bir sonraki katmanına hiçbir veri iletilmez.)

![https://miro.medium.com/max/1612/0*3rRcXavz8hmHXAFL.png](https://miro.medium.com/max/1612/0*3rRcXavz8hmHXAFL.png)

## What is Convolution Neural Network (CNN)?<a name="paragraph5"></a>

A convolutional neural network (ConvNet / Convolutional neural networks -CNN) is a deep learning algorithm that can take an input image and separate various aspects/objects in the image. Convolutional neural networks are deep neural networks that are mainly used to classify images (for example, to name what they see), cluster by similarity (photo search), and perform object recognition in scenes.

- In CNN's, each filter is replicated across the entire visual field. These replicated units share the same parameterization (weight vector and bias) and form a feature map. This means that all the neurons in a given convolutional layer respond to the same feature within their specific response field.

  Weight sharing reduces the training time; this is a direct advantage of the reduction of the number of weight updates that have to take place during backpropagation.
  To reiterate weight sharing occurs when a feature map is generated from the result of the convolution between a filter and input data from a unit within a plane in the convolution layer. All units within this layer plane share the same weights; hence it is called weight/parameter sharing.

  A convolutional neural network learns certain features in images that are useful for classifying the image. Sharing parameters gives the network the ability to look for a given feature everywhere in the image, rather than in just a certain area. This is extremely useful when the object of interest could be anywhere in the image.
  Relaxing the parameter sharing allows the network to look for a given feature only in a specific area. For example, if your training data is of faces that are centered, you could end up with a network that looks for eyes, nose, and mouth in the center of the image, a curve towards the top, and shoulders towards the bottom.
  It is uncommon to have training data where useful features will usually always be in the same area, so this is not seen often.
- CNN captures the **spatial features** from an image. Spatial features refer to the arrangement of pixels and the relationship between them in an image. They help us in identifying the object accurately, the location of an object, as well as its relation with other objects in an image.

  (Bir evrişimsel sinir ağı (ConvNet / Convolutional neural networks -CNN), bir girdi görüntüsünü alıp, görüntüdeki çeşitli görünüşleri/nesneleri birbirinden ayırabilen derin öğrenme algoritmasıdır. Evrişimli sinir ağları, temel olarak görüntüleri sınıflandırmak (örneğin gördüklerini isimlendirmek), benzerlikle kümelemek (fotoğraf arama) ve sahnelerde nesne tanıma yapmak için kullanılan derin yapay sinir ağlarıdır.

- CNN'lerde, her filtre tüm görsel alan boyunca çoğaltılır. Bu çoğaltılmış birimler aynı parametreleştirmeyi (ağırlık vektörü ve sapma) paylaşır ve bir özellik haritası oluşturur. Bu, belirli bir evrişim katmanındaki tüm nöronların, kendi özel yanıt alanlarında aynı özelliğe yanıt verdiği anlamına gelir.
  Ağırlık paylaşımı eğitim süresini azaltır; bu, geri yayılım sırasında gerçekleşmesi gereken ağırlık güncellemelerinin sayısının azaltılmasının doğrudan bir avantajıdır.
  Ağırlık paylaşımını tekrarlamak, evrişim katmanındaki bir düzlem içindeki bir birimden gelen bir filtre ile girdi verileri arasındaki evrişimin sonucundan bir özellik haritası oluşturulduğunda meydana gelir. Bu katman düzlemindeki tüm birimler aynı ağırlıkları paylaşır; dolayısıyla buna ağırlık/parametre paylaşımı denir.

  Bir evrişimli sinir ağı, görüntülerde görüntüyü sınıflandırmak için yararlı olan belirli özellikleri öğrenir. Parametrelerin paylaşılması, ağa belirli bir alanda değil, görüntünün her yerinde belirli bir özelliği arama yeteneği verir. Bu, ilgilenilen nesne görüntünün herhangi bir yerinde olabileceği durumlarda son derece kullanışlıdır.

  Parametre paylaşımının gevşetilmesi, ağın belirli bir özelliği yalnızca belirli bir alanda aramasını sağlar. Örneğin, egzersiz verileriniz ortalanmış yüzlere aitse, görüntünün merkezinde gözleri, burnu ve ağzı, yukarıya doğru bir eğriyi ve alta doğru omuzları arayan bir ağ elde edebilirsiniz.
  Yararlı özelliklerin genellikle her zaman aynı alanda olacağı eğitim verilerine sahip olmak nadirdir, bu nedenle bu sık görülmez.
- CNN bir görüntüden **uzamsal özellikleri** yakalar . Uzamsal özellikler, bir görüntüdeki piksellerin düzenini ve aralarındaki ilişkiyi ifade eder. Nesneyi, bir nesnenin konumunu ve bir görüntüdeki diğer nesnelerle ilişkisini doğru bir şekilde tanımlamamıza yardımcı olurlar.)

![https://www.researchgate.net/publication/326152216/figure/fig2/AS:644307941879809@1530626392505/Deep-convolutional-neural-network-DCNN-architecture-A-schematic-diagram-of-AlexNet.png](https://www.researchgate.net/publication/326152216/figure/fig2/AS:644307941879809@1530626392505/Deep-convolutional-neural-network-DCNN-architecture-A-schematic-diagram-of-AlexNet.png)

## What is segmentation task in NN?<a name="paragraph6"></a>

Image segmentation is the process of classifying each pixel in an image belonging to a certain class and hence can be thought of as a classification problem per pixel. Semantic segmentation falls under supervised learning. There are two types of segmentation techniques;

**Semantic segmentation**<a name="subparagraph3"></a>

Semantic segmentation is the process of classifying each pixel belonging to a particular label. It doesn’t different across different instances of the same object. For example if there are 2 cats in an image, semantic segmentation gives same label to all the pixels of both cats

![https://wiki.tum.de/download/attachments/23561833/sms.png?version=1&modificationDate=1483619907233&api=v2](https://wiki.tum.de/download/attachments/23561833/sms.png?version=1&modificationDate=1483619907233&api=v2)

**Instance segmentation**<a name="subparagraph4"></a>

![https://www.researchgate.net/profile/Vinorth-Varatharasan/publication/339328277/figure/fig1/AS:864554888204291@1583137356688/Semantic-segmentation-left-and-Instance-segmentation-right-8.ppm](https://www.researchgate.net/profile/Vinorth-Varatharasan/publication/339328277/figure/fig1/AS:864554888204291@1583137356688/Semantic-segmentation-left-and-Instance-segmentation-right-8.ppm)

Instance segmentation differs from semantic segmentation in the sense that it gives a unique label to every instance of a particular object in the image. As can be seen in the image above all 5 people are assigned different colours i.e different labels. With semantic segmentation all of them would have been assigned the same colour.

( İmage segmentation, belirli bir sınıfa ait bir görüntüdeki her bir pikseli sınıflandırma işlemidir ve bu nedenle piksel başına bir sınıflandırma problemi olarak düşünülebilir. Semantik segmentasyon denetimli öğrenme kapsamına girer. İki tür segmentasyon tekniği vardır; 

**Semantic segmentation**: Semantik segmentasyon, belirli bir etikete ait her pikseli sınıflandırma işlemidir. Aynı nesnenin farklı örneklerinde farklı değildir. Örneğin bir görüntüde 2 kedi varsa, anlamsal bölümleme her iki kedinin tüm piksellerine aynı etiketi verir. 
**Instance segmentation**: Örnek bölümleme, görüntüdeki belirli bir nesnenin her örneğine benzersiz bir etiket vermesi anlamında anlamsal bölümlemeden farklıdır. Yukarıdaki resimde görüldüğü gibi 5 insana da farklı renkler yani farklı etiketler atanmıştır. Anlamsal bölümleme ile hepsine aynı renk atanırdı.)

## What is classification task in NN? <a name="paragraph7"></a>

Artificial Neural Network applications are generally used in prediction, classification, data association, data interpretation and data filtering. Artificial neural networks used for classification take on the task of classifying input values. Classification is the categorization process in which objects are defined, differentiated and data is understood on the basis of the training set. Classification is a supervised learning technique with a training set and correctly defined observations. All classification tasks depend upon labeled datasets; that is, humans must transfer their knowledge to the dataset in order for a neural network to learn the correlation between labels and data. This is known as supervised learning. 

- Detect faces, identify people in images, recognize facial expressions (angry, joyful)
- Identify objects in images (stop signs, pedestrians, lane markers…)
- Recognize gestures in video
- Detect voices, identify speakers, transcribe speech to text, recognize sentiment invoices
- Classify text as spam (in emails) or fraudulent (in insurance claims); recognize sentiment in text (customer feedback)

(Yapay Sinir Ağları uygulamaları genellikle tahmin, sınıflandırma, veri ilişkilendirme, veri yorumlama ve veri filtreleme işlemlerinde kullanılmaktadır. Sınıflandırma amacıyla kullanılan yapay sinir ağları, girdi değerlerini sınıflama görevini üstlenirler. Sınıflandırma, nesnelerin tanımlandığı, farklılaştığı ve verilerin eğitim seti temelinde anlaşıldığı kategorizasyon sürecidir. Sınıflandırma, bir eğitim seti ve doğru tanımlanmış gözlemlerin bulunduğu denetimli bir öğrenme tekniğidir. Tüm sınıflandırma görevleri, etiketlenmiş veri kümelerine bağlıdır; yani, bir sinir ağının etiketler ve veriler arasındaki ilişkiyi öğrenmesi için insanların bilgilerini veri kümesine aktarması gerekir. Bu, denetimli öğrenme olarak bilinir . 

- Yüzleri algılayın, görüntülerdeki insanları tanımlayın, yüz ifadelerini tanıyın (kızgın, neşeli)
- Görüntülerdeki nesneleri tanımlayın (dur işaretleri, yayalar, şerit işaretleri…)
- Videodaki hareketleri tanıma
- Sesleri algılayın, konuşmacıları tanımlayın, konuşmayı metne dönüştürün, seslerdeki duyguyu       tanıyın
- Metni spam (e-postalarda) veya hileli (sigorta taleplerinde) olarak sınıflandırma; metindeki duyarlılığı tanıma (müşteri geri bildirimi).

![https://learnopencv.com/wp-content/uploads/2017/11/cnn-schema1.jpg](https://learnopencv.com/wp-content/uploads/2017/11/cnn-schema1.jpg)

## Compare segmentation and classification in NN.  <a name="paragraph8"></a>


The difference between segmentation and classification is clear at some extend. And there is a one difference between both of them. The classification process is easier than segmentation, in classification all objects in a single image is grouped or categorized into a single class. While in segmentation each object of a single class in an image is highlighted with different shades to make them recognizable to computer vision.

(Segmentasyon ve sınıflandırma arasındaki fark bir dereceye kadar açıktır. Ve ikisi arasında tek bir fark var. Sınıflandırma işlemi segmentasyondan daha kolaydır, sınıflandırmada tek bir görüntüdeki tüm nesneler tek bir sınıfa gruplanır veya kategorize edilir. Segmentasyon sırasında, bir görüntüdeki tek bir sınıfın her bir nesnesi, bilgisayarla görüye tanınabilmesi için farklı gölgelerle vurgulanır.)

## What is data and dataset difference?  <a name="paragraph9"></a>


- **Data** are observations or measurements (unprocessed or processed) represented as text, numbers, or multimedia.
- A **dataset** is a structured collection of data generally associated with a unique body of work.

## What is the difference between supervised and unsupervised learning in terms of dataset?<a name="paragraph10"></a>


**In a supervised learning** model, the algorithm learns on a labeled **dataset**, providing an answer key that the algorithm can use to evaluate its accuracy on training data. An **unsupervised** model, in contrast, provides unlabeled data that the algorithm tries to make sense of by extracting features and patterns on its own.

Supervised → input and label

Unsupervised →input

(Denetimli bir öğrenme modelinde, algoritma etiketli bir veri kümesi üzerinde öğrenir ve algoritmanın eğitim verileri üzerindeki doğruluğunu değerlendirmek için kullanabileceği bir cevap anahtarı sağlar. Denetimsiz bir model, aksine, algoritmanın kendi başına özellikleri ve kalıpları çıkararak anlamlandırmaya çalıştığı etiketlenmemiş verileri sağlar.)

# Data Preprocessing  <a name="paragraph11"></a>


## Extracting Masks<a name="paragraph12"></a>

### What is color space ?<a name="paragraph13"></a>

A **color space** is a coordinate system in which each color is represented as a single point. Colors are composed of a mixture of blue, green and red colors because they react differently at different wavelengths. 

(**Renk uzayı**, her rengin tek bir nokta olarak temsil edildiği bir koordinat sistemidir. Renkler, farklı dalga boylarında farklı tepkime verdikleri için mavi, yeşil ve kırmızı renklerin karışımından oluşur.)

![https://www.hisour.com/wp-content/uploads/2018/03/RGB-color-space.jpg](https://www.hisour.com/wp-content/uploads/2018/03/RGB-color-space.jpg)

### What RGB stands for ?<a name="paragraph14"></a>

**RGB** means Red, Green, Blue, i.e., the primary colors in additive color synthesis. An **RGB** file consists of composite layers of Red, Green, and Blue, each being coded on 256 levels from 0 to 255. 

( **RGB**, Kırmızı Yeşil Mavi anlamına gelir, yani katkılı renk sentezindeki ana renkler. Bir **RGB** dosyası, her biri 0 ila 255 arasında 256 düzeyde kodlanmış Kırmızı, Yeşil ve Mavi bileşik katmanlarından oluşur.)

### In Python, can we transform from one color space to another?<a name="paragraph15"></a>

There are more than 150 color-space conversion methods available in OpenCV. A popular computer vision library written in C/C++ with bindings for Python, OpenCV provides easy ways of manipulating color spaces.

(OpenCV’de 150’den fazla renk alanı dönüştürme yöntemi mevcuttur. C/C++ ile Python için binding’lerle yazılmış popüler bir bilgisayarlı vision kütüphanesi olan OpenCV, renk uzaylarını değiştirmenin kolay yollarını sağlar.)

### What is the popular library for image processing?<a name="paragraph16"></a>

OpenCV is one of the most famous and widely used open-source **libraries** for computer vision tasks such as **image processing**, object detection, face detection, **image** segmentation, face recognition, and many more. Other than this, it can also be used for machine learning tasks.

(OpenCV, görüntü işleme, nesne algılama, yüz algılama, görüntü bölütleme, yüz tanıma ve daha pek çok bilgisayarla görme görevleri için en ünlü ve yaygın olarak kullanılan açık kaynak kütüphanelerinden biridir. Bunun dışında makine öğrenimi görevleri için de kullanılabilir.)

## Converting into Tensor<a name="paragraph17"></a>

### What is Computational Graph?<a name="paragraph18"></a>

A computational graph is a directed graph where the nodes correspond to **operations** or **variables**. Variables can feed their value into operations, and operations can feed their output into other operations. This way, every node in the graph defines a function of the variables.

(Bir hesaplama grafiği, düğümlerin işlemlere veya değişkenlere karşılık geldiği yönlendirilmiş bir grafiktir. Değişkenler değerlerini işlemlere besleyebilir ve işlemler çıktılarını diğer işlemlere besleyebilir. Bu şekilde, grafikteki her düğüm değişkenlerin bir fonksiyonunu tanımlar.)

![http://media5.datahacker.rs/2021/01/54-1.jpg](http://media5.datahacker.rs/2021/01/54-1.jpg)

### What is Tensor?<a name="paragraph19"></a>

The values fed into the nodes and come out of the nodes are called **tensors**, which is just a fancy word for a multi-dimensional array. Hence, it subsumes scalars, vectors, and matrices as well as tensors of a higher rank.

(Düğümlere beslenen ve düğümlerden çıkan değerlere **tensörler** denir, bu sadece çok boyutlu bir dizi için süslü bir kelimedir.)

![https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg](https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg)

### What is one hot encoding?<a name="paragraph20"></a>

One-Hot Encoding is essentially the representation of categorical variables as binary vectors. These categorical values are first mapped to integer values. Each integer value is then represented as a binary vector with all 0s (except for the integer index marked as 1).

(One-Hot Encoding, temel olarak kategorik değişkenlerin ikili vektörler olarak temsilidir. Bu kategorik değerler ilk önce tamsayı değerlere eşlenir. Her tamsayı değeri daha sonra tüm 0’ları olan bir ikili vektör olarak temsil edilir (1 olarak işaretlenen tamsayı indeksi hariç))

![https://mertmekatronik.com/uploads/images/2020/10/image_750x_5f8c85c715869.jpg](https://mertmekatronik.com/uploads/images/2020/10/image_750x_5f8c85c715869.jpg)

### What is CUDA programming? <a name="paragraph21"></a>

CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on its own GPUs (graphics processing units). CUDA enables developers to speed up compute-intensive applications by harnessing the power of GPUs for the parallelizable part of the computation.

(CUDA , Nvidia tarafından kendi GPU’larında (grafik işlem birimleri) genel hesaplama için geliştirilmiş bir paralel hesaplama platformu ve programlama modelidir. CUDA, geliştiricilerin hesaplamanın paralelleştirilebilir kısmı için GPU’ların gücünden yararlanarak yoğun hesaplama gerektiren uygulamaları hızlandırmalarını sağlar.)

# Design Segmentation Model<a name="paragraph22"></a>

## What is the difference between CNN and Fully CNN (FCNN) ?<a name="paragraph23"></a>

FCNN (Fully Convolutional Neural Network), unlike the classic CNN, which uses the Fully Connected layers after the Convolutional layers in the network, the FCNN can take input of arbitrary size. U-Net is also a network structure based on FCNN. Fully convolutional training takes the M x M image and produces outputs for all sub-images in a single ConvNet forward pass.

## What are the different layers on CNN ?<a name="paragraph24"></a>


Three types of layers construct the CNN: the convolutional layers, pooling layers, and fully connected (FC) layers.

### **1. Convolutional Layer**<a name="paragraph25"></a>

This layer is the first layer that is used to extract the various features from the input images. In this layer, the mathematical operation of convolution is performed between the input image and a filter of a particular size MxM.

(Bu katman, girdi görüntülerinden çeşitli özellikleri çıkarmak için kullanılan ilk katmandır. Bu katmanda, girdi görüntüsü ile belirli bir MxM boyutundaki bir filtre arasında matematiksel evrişim işlemi gerçekleştirilir.)

### **2. Pooling Layer**<a name="paragraph26"></a>

In most cases, a Convolutional Layer is followed by a Pooling Layer. The primary aim of this layer is to decrease the size of the convolved feature map to reduce computational costs. This is performed by decreasing the connections between layers and independently operates on each feature map. Depending upon the method used, there are several types of Pooling operations.

(Çoğu durumda, bir Convolution Katmanı, bir Pooling Katmanı izler. Bu katmanın birincil amacı, hesaplama maliyetlerini azaltmak için convolved feature map’in boyutunu küçültmektir. Bu, katmanlar arasındaki bağlantıları azaltarak gerçekleştirilir ve her bir feature map üzerinde bağımsız olarak çalışır. Kullanılan yönteme bağlı olarak, birkaç tür Pooling işlemi vardır.

### **3. Fully Connected Layer**<a name="paragraph27"></a>

The Fully Connected (FC) layer consists of the weights and biases and the neurons and is used to connect the neurons between two different layers. These layers are usually placed before the output layer and form the last few CNN architecture layers.

(Fully Connected (FC) katmanı, nöronlarla birlikte ağırlıklardan ve biase’lardan oluşur ve nöronları iki farklı katman arasında bağlamak için kullanılır. Bu katmanlar genellikle çıktı katmanından önce yerleştirilir ve bir CNN Mimarisinin son birkaç katmanını oluşturur.)

![https://miro.medium.com/max/3288/1*uAeANQIOQPqWZnnuH-VEyw.jpeg](https://miro.medium.com/max/3288/1*uAeANQIOQPqWZnnuH-VEyw.jpeg)

CNN with Classification Task

## What is activation function ?<a name="paragraph28"></a>

An activation function is a function used in artificial neural networks which output a small value for small inputs and a larger value if its inputs exceed a threshold. If the inputs are large enough, the activation function “fires.” Otherwise, it does nothing. In other words, an activation function is like a gate that checks that an incoming value is greater than a critical number. 
The non-linear functions are known to be the most used activation functions. It makes it easy for a neural network model to adapt to a variety of data and to differentiate between the outcomes.
The purpose of the activation function is to introduce non-linearity into the output of a neuron.

A neural network is essentially just a linear regression model without an activation function. The activation function does the non-linear transformation to the input making it capable to learn and perform more complex tasks.

Many multi-layer neural networks end in a penultimate layer that outputs real-valued scores that are not conveniently scaled and which may be difficult to work with. Here the softmax is very useful because it converts the scores to a normalized probability distribution, which can be displayed to a user or used as input to other systems. For this reason, it is usual to append a softmax function as the final layer of the neural network.

(Birçok çok katmanlı sinir ağı, uygun şekilde ölçeklendirilmemiş ve birlikte çalışması zor olabilecek gerçek değerli puanlar veren sondan bir önceki katmanda sona erer. Burada softmax çok kullanışlıdır çünkü puanları bir kullanıcıya gösterilebilen veya diğer sistemlere girdi olarak kullanılabilen normalleştirilmiş bir olasılık dağılımına dönüştürür . Bu nedenle, sinir ağının son katmanı olarak bir softmax işlevi eklemek olağandır.)

![https://themaverickmeerkat.com/img/softmax/sigmoid_plot.jpg](https://themaverickmeerkat.com/img/softmax/sigmoid_plot.jpg)

Logistic Activation Function

# Train<a name="paragraph29"></a>

## What is parameter and hyper-parameter in NN ?<a name="paragraph30"></a>

A model parameter is a configuration variable internal to the model and whose value can be estimated from data. Parameters are key to machine learning algorithms. They are the part of the model that is learned from historical training data. These are the coefficients of the model, and the model itself chooses them. It means that the algorithm, while learning, optimizes these coefficients (according to a given optimization strategy) and returns an array of parameters that minimize the error. 


**Hyperparameters**: these are elements that, differently from the previous ones, you need to set. Furthermore, the model will not update them according to the optimization strategy: your manual intervention will always be needed.

- Number of hidden layers
- Learning rate
- Momentum
- Activation function
- Minibatch size
- Epochs
- Dropout rate

## Validation Dataset<a name="paragraph31"></a>

The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as a skill on the validation dataset is incorporated into the model configuration. The validation set is used to evaluate a given model, but this is for frequent evaluation. Hence the model occasionally sees this data but never does it “Learn” from this. We use the validation set results, and update higher level hyperparameters. So the validation set affects a model, but only indirectly. The validation set is also known as the Dev set or the Development set. This makes sense since this dataset helps during the “development” stage of the model.

## What is an epoch?<a name="paragraph32"></a>

The number of epochs is a hyperparameter that defines the number of times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. An epoch is comprised of one or more batches.

## What is batch?<a name="paragraph33"></a>

The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

## What is iteration?<a name="paragraph34"></a>

For each complete epoch, we have several iterations. Iteration is the number of batches or steps through partitioned packets of the training data, needed to complete one epoch. if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch. So then it will take x/y iterations to complete 1 epoch.

## What Is the Cost Function?<a name="paragraph35"></a>

It is a function that measures the performance of a Machine Learning model for given data. Cost Function quantifies the error between predicted values. A cost function measures “how good” a neural network did concerning its given training sample and the expected output. It also may depend on variables such as weights and biases.)

(Verilen veriler için bir Makine Öğrenimi modelinin performansını ölçen bir işlevdir. Maliyet Fonksiyonu, tahmin edilen değerler arasındaki hatayı ölçer.Maliyet fonksiyonu, verilen eğitim örneğine ve beklenen çıktıya göre bir sinir ağının “ne kadar iyi” olduğunun bir ölçüsüdür. Ayrıca ağırlıklar ve bias’lar gibi değişkenlere de bağlı olabilir.)

## What is/are the purpose(s) of an optimizer in NN?<a name="paragraph36"></a>

Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to **reduce the losses**. How you should change the weights or learning rates of your neural network to reduce the losses is defined by the optimizers you use. Optimization algorithms are responsible for reducing the losses and providing the most accurate results possible.

(Optimize ediciler, kayıpları azaltmak için sinir ağının ağırlıklar ve öğrenme oranı gibi özelliklerini değiştirmek için kullanılan algoritmalar veya yöntemlerdir.Kayıpları azaltmak için sinir ağınızın ağırlıklarını veya öğrenme oranlarını nasıl değiştirmeniz gerektiği, kullandığınız optimize ediciler tarafından belirlenir. Optimizasyon algoritmaları, kayıpları azaltmaktan ve mümkün olan en doğru sonuçları sağlamaktan sorumludur.)

## What is Batch Gradient Descent & Stochastic Gradient Descent?<a name="paragraph37"></a>

In **Batch Gradient Descent**, all the training data is taken into consideration to take a single step. We take the average of the gradients of all the training examples and then use that mean gradient to update our parameters. So that’s just one step of gradient descent in one epoch. In **Stochastic Gradient Descent** (SGD), we consider just one example at a time to take a single step. We do the following steps in **one epoch** for SGD: 

1. Take an example 

2. Feed it to Neural Network 

3. Calculate its gradient 

4. Use the gradient we calculated in step 3 to update the weights 

5. Repeat steps 1–4 for all the examples in training dataset

SGD can be used for larger datasets. It converges faster when the dataset is large as it causes updates to the parameters more frequently. 

(SGD, daha büyük veri kümeleri için kullanılabilir. Parametrelerin daha sık güncellenmesine neden olduğu için veri kümesi büyük olduğunda daha hızlı yakınsar.)

## What is Backpropogation ? What is used for ?<a name="paragraph38"></a>

Artificial neural networks use backpropagation as a learning algorithm to compute a gradient descent for weights. Desired outputs are compared to achieved system outputs, and then the systems are tuned by adjusting connection weights to narrow the difference between the two as much as possible. The algorithm gets its name because the weights are updated backward, from output towards input. A neural network propagates the signal of the input data forward through its parameters towards the moment of decision and then *backpropagates* information about the error, in reverse through the network, so that it can alter the parameters. This happens step by step: 

1. The network guesses data, using its parameters
2. The network is measured with a loss function
3. The error is backpropagated to adjust the wrong-headed parameters

Backpropagation takes the error associated with a wrong guess by a neural network and uses that error to adjust the neural network’s parameters in the direction of less error.

![https://machinelearningknowledge.ai/wp-content/uploads/2019/10/Backpropagation.gif](https://machinelearningknowledge.ai/wp-content/uploads/2019/10/Backpropagation.gif)
## Object Recognition<a name="anabaslik2"></a>
### What is an image classification task?<a name="paragraph39"></a>
It involves the extraction of information from an image and then associating the extracted information to one or more class labels. Image classification within the machine learning domain can be approached as a supervised learning task.

Image classification, at its very core, is the task of assigning a label to an image from a predefined set of categories. Practically, this means that our task is to analyze an input image and return a label that categorizes the image. The label is always from a predefined set of possible categories.

For example, let’s assume that our set of possible categories includes:

                   categories = {cat,dog,panda}
    
Then we present the following image to our classification system:
![enter image description here](https://www.proutletplus.com/wp-content/uploads/2019/11/k%C3%B6%C3%B6%C3%B6.jpg)

Our goal here is to take this input image and assign a label to it from our **categories set** — in this case, **dog**.

Our classification system could also assign multiple labels to the image via probabilities, such as dog: 95%; cat: 4%; panda: 1%.

More formally, given our input image of W×H pixels with three channels, Red, Green, and Blue, respectively, our goal is to take the W×H×3 = N pixel image and figure out how to correctly classify the contents of the image.

-   **Image Classification**: Predict the type or class of an object in an image.
    -   _Input_: An image with a single object, such as a photograph.
    -   _Output_: A class label (e.g. one or more integers that are mapped to class labels).
   
### What is an object localization task?<a name="paragraph40"></a>
Image classification involves predicting the class of one object in an image. Object localization refers to identifying the location of one or more objects in an image and drawing bounding boxes around their extent. Object detection combines these two tasks and localizes and classifies one or more objects in an image.

(Image classification(Görüntü sınıflandırma), bir görüntüdeki bir nesnenin sınıfını tahmin etmeyi içerir. Object localization(Nesne yerelleştirme), bir görüntüdeki bir veya daha fazla nesnenin konumunu belirleme ve boyutlarının etrafına bol miktarda kutu çizme anlamına gelir. Object detection(Nesne algılama) bu iki görevi birleştirir ve bir görüntüdeki bir veya daha fazla nesneyi yerelleştirir ve sınıflandırır.)

The task of object localization is to predict the object in an image as well as its boundaries. The difference between object localization and object detection is subtle. Simply, object localization aims to locate the main (or most visible) object in an image while object detection tries to find out all the objects and their boundaries.

(Nesne lokalizasyonunun görevi, bir görüntüdeki nesneyi ve sınırlarını tahmin etmektir. Nesne yerelleştirme ve nesne algılama arasındaki fark çok incedir. Basitçe, nesne yerelleştirme bir görüntüdeki ana (veya en görünür) nesneyi bulmayı amaçlarken nesne algılama tüm nesneleri ve sınırlarını bulmaya çalışır.)

-   **Object Localization**: Locate the presence of objects in an image and indicate their location with a bounding box.
    -   _Input_: An image with one or more objects, such as a photograph.
    -   _Output_: One or more bounding boxes (e.g. defined by a point, width, and height).
 
### What is an object detection task? <a name="paragraph41"></a>
Object detection is a computer vision technique that identifies and locates objects within an image or video. Specifically, object detection draws bounding boxes around these detected objects, which allow us to locate where said objects are in (or how they move through) a given scene.

Object detection is commonly confused with image recognition, so before we proceed, we must clarify the distinctions between them.

Image recognition assigns a label to an image. A picture of a dog receives the label “dog”. A picture of two dogs still receives the label “dog”. On the other hand, object detection draws a box around each dog and labels the box “dog”. The model predicts where each object is and what label should be applied. In that way, object detection provides more information about an image than recognition.

-   **Object Detection**: Locate the presence of objects with a bounding box and types or classes of the located objects in an image.
    -   _Input_: An image with one or more objects, such as a photograph.
    -   _Output_: One or more bounding boxes (e.g. defined by a point, width, and height), and a class label for each bounding box.
   ![enter image description here](https://www.fritz.ai/images/object_detection_vs_image_recognition.jpg)
 ![enter image description here](https://appsilon.com/assets/uploads/2018/08/types.png)
### What is an object recognition task?<a name="paragraph42"></a>
Object recognition is a general term to describe a collection of related computer vision tasks that involve identifying objects in digital photographs.

We will be using the term object recognition broadly to encompass both image classification (a task requiring an algorithm to determine what object classes are present in the image) as well as to object detection (a task requiring an algorithm to localize all objects present in the image.)

Object Recognition is responding to the question "What is the object in the image" Whereas, Object detection is answering the question "Where is that object"?

(Nesne Tanıma, "Görüntüdeki nesne nedir" sorusuna yanıt verirken Nesne algılama, "O nesne nerede" sorusuna yanıt verir)

-   _input_: an image containing unknown object(s)
    
    Possibly, the position of the object can be marked in the input, or the input might be only a clear image of (not-occluded) object.
    
-   _output_: position(s) and label(s) (names) of the objects in the image.

<p  align="center"><img  src="images/Object_Recognition"  width=""> </p>
### What is bounding box regression?<a name="paragraph43"></a>
Bounding-box regression is a popular technique to refine or predict localization boxes in recent object detection approaches. Typically, bounding-box regressors are trained to regress from either region proposals or fixed anchor boxes to nearby bounding boxes of predefined target object classes.

### What is non-max suppression?<a name="paragraph44"></a>
Non Maximum Suppression (NMS) is a technique used in many computer vision algorithms. It is a class of algorithms to select one entity (e.g. bounding boxes) out of many overlapping entities. The selection criteria can be chosen to arrive at particular results. Most commonly, the criteria are some form of probability number along with some form of overlap measure (e.g. IOU).

Most object detection algorithms use NMS to whittle down a large number of detected rectangles to a few. At the most basic level, most object detectors do some form of windowing. Many, thousands, windows of various sizes and shapes are generated either directly on the image or on a feature of the image. These windows supposedly contain only one object, and a classifier is used to obtain a probability/score for each class. Once the detector outputs a large number of bounding boxes, it is necessary to pick the best ones. NMS is the most commonly used algorithm for this task. In essence, it is a form of clustering algorithm.

(Çoğu nesne algılama algoritması, çok sayıda algılanan dikdörtgeni birkaç taneye indirgemek için NMS kullanır. En temel düzeyde, çoğu nesne dedektörü bir tür pencereleme yapar. Çeşitli boyut ve şekillerde çok sayıda, binlerce pencere ya doğrudan görüntü üzerinde ya da görüntünün bir özelliği üzerinde oluşturulur. Bu pencerelerin yalnızca bir nesne içerdiği varsayılır ve her sınıf için bir olasılık/puan elde etmek için bir sınıflandırıcı kullanılır. Dedektör çok sayıda sınırlayıcı kutu çıkardığında, en iyilerini seçmek gerekir. NMS, bu görev için en yaygın kullanılan algoritmadır. Özünde, kümeleme algoritmasının bir şeklidir.)

![enter image description here](https://appsilon.com/assets/uploads/2018/08/nonmax-1.png)
