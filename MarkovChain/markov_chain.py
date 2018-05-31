from enum import Enum

import numpy as np


class State(Enum):
    SLEEP = 0,
    RUN = 1,
    ICECREAM = 2


class TS:
    """Transition to next state"""

    def __init__(self, next_state: State, probability: float):
        self.next_state = next_state
        self.probability = probability


transition_probabilities = {
    State.SLEEP: [TS(State.SLEEP, 0.2), TS(State.RUN, 0.6), TS(State.ICECREAM, 0.2)],
    State.RUN: [TS(State.SLEEP, 0.1), TS(State.RUN, 0.6), TS(State.ICECREAM, 0.3)],
    State.ICECREAM: [TS(State.SLEEP, 0.2), TS(State.RUN, 0.7), TS(State.ICECREAM, 0.1)]
}


def activity_forecast(days: int):
    activity_today = State.SLEEP  # start
    activity_list = [activity_today]  # store the sequence of states taken
    prob = 1  # start probability is 1
    for i in range(0, days):
        transitions = transition_probabilities[activity_today]  # select next state transition probabilities
        # choose between possible next transitions with their probabilities
        change: TS = np.random.choice(transitions, replace=True, p=[p.probability for p in transitions])

        prob *= change.probability  # multiply total prob with choice
        activity_list.append(change.next_state)  # add state to the list of visited states
        activity_today = change.next_state  # set today to next activity

    print_result(activity_list, activity_today, prob, days)


def print_result(activity_list, activity_today, prob, days):
    print("Possible states: " + str([i.name for i in activity_list]))
    print("End state after " + str(days) + " days: " + activity_today.name)
    print("Probability of the possible sequence of states: " + str(prob))


if __name__ == '__main__':
    activity_forecast(5)
