def generate_commentary(score, wickets, run_rate, probability):

    if probability > 0.75:

        return (
            "The batting team is in a dominant position. "
            "A strong scoring rate combined with controlled wicket loss "
            "indicates high chances of securing victory."
        )

    elif probability > 0.60:

        return (
            "The match momentum slightly favours the batting side. "
            "Maintaining the run rate while avoiding unnecessary wickets "
            "will significantly increase winning probability."
        )

    elif probability > 0.50:

        return (
            "The match is evenly balanced. Tactical shot selection "
            "and strike rotation will be crucial to build pressure "
            "on the opposition."
        )

    else:

        return (
            "The batting side is currently under pressure. "
            "Increasing scoring momentum while protecting wickets "
            "is essential to turn the match situation around."
        )