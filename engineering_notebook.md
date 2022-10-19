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