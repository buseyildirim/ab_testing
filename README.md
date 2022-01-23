# AB Testing
A/B testing is basically a statistical test in which two or more variants of an approach are randomly shown to users and used to determine which variation performs better for a given conversion goal. There may be a mathematical difference between the variants, but this mathematical difference may have occurred by chance. To test this, the ab test is applied. It allows us to test whether this difference occurs by chance at a given confidence interval.

## Business Problem : <br>
Company X has recently introduced a new type of bidding, average bidding, as an alternative to the current type of bidding called maximum bidding.
One of our clients, bombabomba.com, decided to test this new feature and wants to do an A/B test to see if average bidding converts more than maximum bidding.

## Dataset Story:

In this dataset, which includes the website information of bombabomba.com, there is information such as the number of advertisements that users see and click, as well as earnings information from here.<br>

There are two separate data sets, the control and test groups.<br>

The max binding strategy was presented to the control group, and the average binding strategy was presented to the test group.

<br>

## Variables

<table>
  <tr >
    <th>Variable</th>
    <th>Description</th> 
  </tr>
    <tr>
    <td>Impression</td>
    <td>TAd views count</td> 
  </tr>
  
  <tr>
    <td>Click</td>
    <td>The number of clicks on the displayed ad</td> 
  </tr>
  <tr>
    <td>Purchase</td>
    <td>The number of products purchased after the ads clicked</td> 
  </tr>
    <td>Earning</td>
    <td>Earnings after purchased products</td> 
  </tr>
</table>

