## Repo for our submission to NUS AI Innovation challenge 2023. We were one of the 8 finalists selected out of 104 teams.

- The competition required the contestants to establish the stock timing model, spontaneously find the best trading opportunity to complete the trading and strive for the lowest overall trading cost of the stock.
- The competition offered 500 stocks, each stock must complete buy and sell of 100 shares a day, and each trading of the number of shares can be distributed freely.
- The trading time of each stock is from 9:30 to 11:30 and 13:00 to 15:00 daily.
- Contestants needed to select several optimal time points for each stock to trade within the trading time. "Buy low, Sell high".

After thorough literature review, We zeroed in on using Deep Re-inforcement learning method called PPO (Proximal Policy Optimizer).
<img width="326" alt="image" src="https://user-images.githubusercontent.com/93938450/235061302-81cc709d-5d89-459b-984c-39715e910e28.png">

As per the innovation, we brought to the competition, we introduced the following:
- Custom Reward function.

- Relevant State variables.

- Short-selling.

- Explainability of results.


<img width="327" alt="image" src="https://user-images.githubusercontent.com/93938450/235061516-7a55dd36-e961-48c5-a2fc-c58a61af21fd.png">

