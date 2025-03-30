# Function to assess financial risk
def assess_risk(current_price, predicted_price):
    change = ((predicted_price - current_price) / current_price) * 100
    if change > 5:
        return "🟢 Low Risk (Expected Growth)"
    elif -5 <= change <= 5:
        return "🟡 Medium Risk (Stable)"
    else:
        return "🔴 High Risk (Possible Decline)"
