### Oct 18 2022, Tuesday

1.

To recap the current situation: I've just gotten the SNC algorithm to work on caterpillar. I was able to train within a radius of 2 kpc (without filtering small clusters), and using a knn of 100 to construct the clustering graph. The accuracy achieved wasn't the greatest but was bearable. 

My current undertaking is to expand the radius to 5kpc and perform small cluster filtering of stars that are within the said radius. A dataset with those parameters have a cardinality between $10^4$ and $10^5$. That explodes the memory when knn=100. So, I cut knn down to 10, and it works fine. 

The training results, currently, seem pretty bleak. The egnn algorithm is not getting high std (expected) but also very low accuracy of classification. The two rows on the countmatrix are the actual labels (so, we can see that they are fairly even, which is surprising). The columns are the predicted labels. 

![image-20221018113000277](assets/83b102de9e6ce4f095b82367d35fb676c3fe6d07.png)

It seems to be generating only 1 cluster prediction, which is reasonable since most of the edges are predicted to be connected. Sometimes the first row has 3x as much--> too many connections which makes sense since we reduce knn from 100 to 10. 

The LR is effective, but it is not taking away a lot from each episode of training (we always start from a high loss). 

![image-20221018122453062](assets/426b9d3f604fb62c3dbe575f21d0137905379109.png)

Matters hardly improve over time. 

![image-20221018130704054](assets/60ba01b77da548611981f5aa31f0f09e6bbce645.png)

2.

I'm upping the knn to 30 and the egnn_regularizer term to 0.3 from 0.1 

It doesn't work.

3.

It looks like we will have to use sampling anyhow, the reason is that if we don't, we cannot account for the difference in density between datasets. So the learning done on caterpillar cannot be transferred to Gaia. So the idea may be to use sampling to obtain a bunch of clustering results, and then use those results to construct the final result for the dataset's clustering. 

So here is 10000 sampled, 100 knn result. It seems to do a better job with both clustering and the egnn step (as in, it's closer to 0.5)

![](assets/2022-10-18-21-41-27-image.png)

![](assets/2022-10-18-21-43-39-image.png)

It seems clustering accuracy decreases as we train further, while the egnn loss keeps going down. 

![](assets/2022-10-18-22-11-55-image.png)

![](assets/2022-10-18-22-12-22-image.png)

I will investigate why. Possibly it is overfitting to the train dataset? I will print the egnn loss for the validation dataset.

Overfitting is indeed occuring,

![](assets/2022-10-18-23-02-21-image.png)

It seems that we eventually learn to enlarge the average predicted probability value (because some are indeed linkages separate). However, on the validation side, we never quite learn that. 

Here's the result with large regularization:

![](assets/2022-10-18-23-41-39-image.png)

4.

After some considerations, I've determined that some changes need to be made. 

First, new testing has shown that simply increasing the regularizer (that enforces std) is not sufficient to create good clustering results (in fact, it's somehow forcing the clusterer to group all points into a single cluster)

Second, a major shortcoming of the current system is its knn graph being shortsighted. For each evalution (convolution) of the clusterer, we are looking at a small patch of points that are close together. If you have a smooth surface, it is very difficult to determine where to cut the surface so as to separate into clusters! What we need are global viewpoints. The graph convolution with a global viewpoint will function more like an attention mechanism than a normal convolution system. There are many ways to construct global viewpoints-->small sample dense graphs, randomly chosen edges, randomly chosen edges with weighted probability based on distance to the origin point. knn + randomly chosen edges. It's difficult to determine which one would work based on intuitions.

Third, we need to find a way to generate cluster assignments for all points based on few known points or edges. A simple way is to do find the nearest neighbor of each point, and just set it to be the same cluster. A more sophisticated way is to construct a sparse knn graph based on all points. Predict edges for these knn edges and then use the same approach as before. 

5.

I actually have been reading the axis wrong. The top row represents different. Random graph connection is bad. Local view actually isn't much of an issue because of how phase-mixed the clusters are. 

6.

Now I will try the old knn=100 setting, without regularization and using a linear loss. Other losses, which are designed to reward pushing things to extreme, causes everything to be pushed to an extreme from the beginning (the same extreme). We must, unfortunately, conclude that the dataset is simply not containing sufficient info for clear separation of edges. 

7.

We can (potentially) improve the cluster generation step by introducing some spectral component. But the egnn is very limited. So, I've opted to do another clustering algo using projection + kmeans instead. 

This has its own superweird problem: the loss keeps decreasing, but the acc goes down.

![](assets/2022-10-19-09-40-13-image.png)

![](assets/2022-10-19-09-41-05-image.png)

![](assets/2022-10-19-09-43-09-image.png)

Weird things happen partly because what you entered is not probability prediction. It's probability density.

### November 5th, 2022 Saturday

1.

Kmeans's naive loss using the distance is not going to work out. The reason is simply that the model will project every point into a very tiny compact space--thus all the distance costs are very small.

2.

There is a critical mistake in the loss function I employed for GMM. The loss matching previously doesn't work because of two reasons. 1. the labels of the dataset do not start at 0 or 1 ---> these are the labels corresponding to clusters with a size larger than a certain cutoff number which we set in the `sample_space` function. 2. when matching a bunch of predicted clusters with actual clusters, if the number of predicted cluster < # of actual cluster, then only a small percent of the actual clusters are matched up. This is not idea, because the cost associated with sampling a bunch of points is not considered. In the ideal case, each predicted cluster should be allowed to match up with multiple actual clusters. Actually, that wouldn't make sense either... 

On second thought, cluster matching simply doesn't seem wise. Actually, it seems alright as long as every single real cluster is matched up with something. 

4.

Training results of GNN_GMM:

![](assets/2022-11-06-01-18-55-image.png)

![](assets/2022-11-06-01-19-08-image.png)

Projection results of GMM after 1 epoch:

![](assets/2022-11-06-01-17-43-image.png)

![](assets/2022-11-06-01-18-18-image.png)

After 5 epochs:

![](assets/2022-11-06-01-21-36-image.png)

![](assets/2022-11-06-01-21-56-image.png)

I think it is very clear that we are not learning. We need to add more features! There's no other way around. 

### November 6th, 2022 Sunday

1.

I used k-means clustering to be the new clustering method for GNN and arrives at a very weird training result:

![](assets/2022-11-06-21-19-26-image.png)

![](assets/2022-11-06-21-19-47-image.png)

It's due to a weird error:

![](assets/2022-11-06-21-21-58-image.png)

Turns out it's just learning to project everything onto a single fricking point:

![](assets/2022-11-06-21-33-05-image.png)

So, I think we can conclude pretty clearly that projection-based methods are no better at the task. 

2.

This is the egnn results after adding new variables like x,y,z and v. It is much better. There's a high precision, but I think once we trade precision for recall, it could generate decent results. 

![](assets/2022-11-07-02-23-05-image.png)

![](assets/2022-11-07-02-26-21-image.png)

### November 21st, 2022 Monday

The new goal of the game is to first make a lot of plots to gain a deep understanding into the dataset. And then, we try to reproduce Kaley's work and using our algorithms with Kaley's settings. 

9:00 PM, currently, I've just copied over the new dataset with velocity and locations and verified the validity of the data. Data are not very separable for heavy substructures. For ultra-faint dwarf galaxy however, one of the top 10 largest clusters is separable, but with the other stars, it might be hard.

### December 15th, 2022 Thursday

GNN based SNC seems to perform worse than GMM. I run the SNC with knn=10, and n_cluster=50. It achieves low-mid recall and relatively good precision. Then I increased knn=30 and n_cluster=100. It achieves high-mid recall and mid precision. 

![Screenshot from 2022-12-15 00-01-59.png](assets/4a00e598ea821d28b7cfa837e05873e03bfa7ed1.png)

The red one represents knn=10 and n_cluster=50. The yellow one is knn=30, n_cluster=100. Here are the two in the same order.

![run with n=50, knn=10.png](assets/f54725fca1523cf7665605d36501c26d9824e411.png)

![Screenshot from 2022-12-15 00-13-15.png](assets/7b8ba0418520718fabfaedfbf11961ff495aeddf.png)
