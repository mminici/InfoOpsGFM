# Coordinated Activity on Twitter

In recent years, social networks have been leveraged in order to develop influence campaigns. 
These deceptive and manipulative efforts seek to sway public opinion by disseminating fabricated and misleading information, usually with the intent of favoring a particular political side or ideology. 
They take advantage of the discussion that naturally develops around such events to introduce misinformation and target common users.

Rather than being orchestrated by a solitary individual, influence campaigns tend to be sophisticated and coordinated endeavors undertaken by groups of malicious users operating in unison. These nefarious actors exploit a wide array of techniques and tactics.
There is an extended body of research that focuses on the detection of users belonging to influence campaigns.

This repository holds the implementation of state-of-the-art detection methods focused on building similarity networks of users based on different suspicious behavioural traces:
- Fast Retweet
- Co-Retweet
- Co-URL
- Hashtag Sequence
- Text Similarity

Additionally, you can find a Co-sharing implementation which generalizes the five methods and can potentially work for any kind of feature that a user can share.

The state-of-art methods were used all together to have a comprehensive overview of users similarity across multiple features. To do this, the utils script provides a function that allows to merge multiple similarity networks. In the same script, you can also find the function for the computation of centrality properties in a given network - needed in our method to identify coordinated users.

```
@inproceedings{luceri2024unmasking,
    author={Luca Luceri and Valeria Pantè and Keith Burghardt and Emilio Ferrara},
    title = {Unmasking the web of deceit: Uncovering coordinated activity to expose information operations on twitter},
    booktitle = {Proceedings of the 2024 ACM Web Conference},
    year = {2024}
}
```

[Here](https://arxiv.org/abs/2310.09884) you can find the pre-print of our paper.
