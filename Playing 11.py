
#Step 1: Import necessary libraries and dataframes

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime as dt


# Use the existing dataframes directly
batting_df = pd.read_csv('batsman_scorecard.csv')
bowlers_df = pd.read_csv('bowler_scorecard.csv')
# matchlevel_df = matchlevel_scorecard

#Step 2: Initialize data structures for batsman and bowler details

non_dismissals = []

batsmanid_to_details=dict()
#Step 3: Iterate through each row in batting_df and update batsmanid_to_details
for index,row in batting_df.iterrows():
    if row["batsman_id"] not in batsmanid_to_details:
        batsmanid_to_details[row["batsman_id"]]=dict()
        batsmanid_to_details[row["batsman_id"]]["name"]=row["batsman"]
        batsmanid_to_details[row["batsman_id"]]["matches"]=dict()
        batsmanid_to_details[row["batsman_id"]]["50s"]=0
        batsmanid_to_details[row["batsman_id"]]["100s"]=0
        batsmanid_to_details[row["batsman_id"]]["dismissals"]=0
        batsmanid_to_details[row["batsman_id"]]["runs"]=0
        batsmanid_to_details[row["batsman_id"]]["balls"]=0
        batsmanid_to_details[row["batsman_id"]]["wicket_keeper"]=0
    batsmanid_to_details[row["batsman_id"]]["50s"]+= 1 if row["runs"]>=50 else 0
    batsmanid_to_details[row["batsman_id"]]["100s"]+= 1 if row["runs"]>=100 else 0
    batsmanid_to_details[row["batsman_id"]]["runs"]+= row["runs"]
    batsmanid_to_details[row["batsman_id"]]["dismissals"]+= 0 if row["wicket kind"] in non_dismissals else 1
    batsmanid_to_details[row["batsman_id"]]["matches"][row["match id"]]=dict()
    batsmanid_to_details[row["batsman_id"]]["matches"][row["match id"]]["date"]=row["match_dt"]
    batsmanid_to_details[row["batsman_id"]]["matches"][row["match id"]]["strike_rate"]=row["strike_rate"]
    try:
        batsmanid_to_details[row["batsman_id"]]["matches"][row["match id"]]["average"]=batsmanid_to_details[row["batsman_id"]]["runs"]/batsmanid_to_details[row["batsman_id"]]["dismissals"]
    except:
        batsmanid_to_details[row["batsman_id"]]["matches"][row["match id"]]["average"]=batsmanid_to_details[row["batsman_id"]]["runs"]
    batsmanid_to_details[row["batsman_id"]]["matches"][row["match id"]]["50s"]=batsmanid_to_details[row["batsman_id"]]["50s"]
    batsmanid_to_details[row["batsman_id"]]["matches"][row["match id"]]["100s"]=batsmanid_to_details[row["batsman_id"]]["100s"]
    batsmanid_to_details[row["batsman_id"]]["balls"]+=row["balls_faced"]
    batsmanid_to_details[row["batsman_id"]]["wicket_keeper"]+=row["is_batsman_keeper"]

#Step 4: Iterate through each row in bowlers_df and update bowlerid_to_details
bowlerid_to_details=dict()
for index,row in bowlers_df.iterrows():
    if row["bowler_id"] not in bowlerid_to_details:
        bowlerid_to_details[row["bowler_id"]]=dict()
        bowlerid_to_details[row["bowler_id"]]["name"]=row["bowler"]
        bowlerid_to_details[row["bowler_id"]]["matches"]=dict()
        bowlerid_to_details[row["bowler_id"]]["wickets"]=0
        bowlerid_to_details[row["bowler_id"]]["runs"]=0
        bowlerid_to_details[row["bowler_id"]]["balls"]=0
        bowlerid_to_details[row["bowler_id"]]["4w"]=0
    bowlerid_to_details[row["bowler_id"]]["matches"][row["match id"]]=dict()
    bowlerid_to_details[row["bowler_id"]]["matches"][row["match id"]]["date"]=row["match_dt"]
    bowlerid_to_details[row["bowler_id"]]["matches"][row["match id"]]["balls"]=row["balls_bowled"]
    bowlerid_to_details[row["bowler_id"]]["matches"][row["match id"]]["runs"]=row["runs"]
    bowlerid_to_details[row["bowler_id"]]["matches"][row["match id"]]["wickets"]=row["wicket_count"]
    bowlerid_to_details[row["bowler_id"]]["matches"][row["match id"]]["4w"]=(row["wicket_count"])//(4*row["inning"])
    bowlerid_to_details[row["bowler_id"]]["4w"]+=(row["wicket_count"])//(4*row["inning"])
    bowlerid_to_details[row["bowler_id"]]["balls"]+=row["balls_bowled"]
    bowlerid_to_details[row["bowler_id"]]["runs"]+=row["runs"]
    bowlerid_to_details[row["bowler_id"]]["wickets"]+=row["wicket_count"]

#Step 5: Function to calculate form factors
def get_form_factors(ppg,i):
    l=list(ppg[i].values())
    if "batting_score" in ppg[i]:
        l.remove(ppg[i]["batting_score"])
    if "bowling_score" in ppg[i]:
        l.remove(ppg[i]["bowling_score"])
    points=sorted(l)
    dates=[x[0] for x in points]
    scores=[x[1] for x in points]
    dates = pd.to_datetime(dates)
    most_recent_date = dates.max()
    time_diffs = (most_recent_date - dates).days
    decay_rate = 0.1
    weights = np.exp(-decay_rate * time_diffs)
    weighted_average = np.sum(weights * scores) / np.sum(weights)
    normal_std_dev = np.std(scores)
    return [weighted_average,normal_std_dev]

#Step 6: Function to calculate rankings
def get_rankings(ppg,max_d,type):
    rankings=[]
    for i in ppg:
        c_score=0
        if "batting_score" in ppg[i]:
            c_score+=ppg[i]["batting_score"]
        if "bowling_score" in ppg[i]:
            c_score+=ppg[i]["bowling_score"]
        w_av,std_dev=get_form_factors(ppg,i)
        w_av/=max_d[type][0]*0.01
        std_dev/=max_d[type][1]*0.01
        c_score/=max_d[type][2]*0.01
        final_score=0.7*c_score+0.2*w_av+0.1*std_dev
        rankings.append([float(final_score),i])
    return sorted(rankings,reverse=True)

#Step 7: Evaluate batsman performance

def evaluate_batsman(batsmanid_to_details,batsman_ppg,batsman_id):
    if batsman_id not in batsman_ppg:
        batsman_ppg[batsman_id]=dict()
    for match_id in batsmanid_to_details[batsman_id]["matches"]:
        score=0
        if batsmanid_to_details[batsman_id]["matches"][match_id]["strike_rate"]>=150:
            score+=50
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["strike_rate"]>=100:
            score+=40
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["strike_rate"]>=80:
            score+=30
        if batsmanid_to_details[batsman_id]["matches"][match_id]["average"]>=50:
            score+=30
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["average"]>=40:
            score+=20
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["average"]>=30:
            score+=10
        else:
            score+=5
        if batsmanid_to_details[batsman_id]["matches"][match_id]["100s"]>=3:
            score+=30
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["100s"]==2:
            score+=20
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["100s"]==1:
            score+=10
        if batsmanid_to_details[batsman_id]["matches"][match_id]["50s"]>=5:
            score+=20
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["50s"]>=3:
            score+=10
        elif batsmanid_to_details[batsman_id]["matches"][match_id]["50s"]>=1:
            score+=5
        if match_id not in batsman_ppg[batsman_id]:
            batsman_ppg[batsman_id][match_id]=[batsmanid_to_details[batsman_id]["matches"][match_id]["date"],score]
        batsman_ppg[batsman_id][match_id][1]+=score
    c_score=0
    strike_rate=batsmanid_to_details[batsman_id]["runs"]/batsmanid_to_details[batsman_id]["balls"]
    avg=batsmanid_to_details[batsman_id]["runs"]/batsmanid_to_details[batsman_id]["dismissals"]
    fifty=batsmanid_to_details[batsman_id]["50s"]
    hundred=batsmanid_to_details[batsman_id]["100s"]
    if strike_rate>=150:
        c_score+=50
    elif strike_rate>=100:
        c_score+=40
    elif strike_rate>=80:
        c_score+=30
    if avg>=50:
        c_score+=30
    elif avg>=40:
        c_score+=20
    elif avg>=30:
        c_score+=10
    else:
        c_score+=5
    if hundred>=3:
        c_score+=30
    elif hundred==2:
        c_score+=20
    elif hundred==1:
        c_score+=10
    if fifty>=5:
        c_score+=20
    elif fifty>=3:
        c_score+=10
    elif fifty>=1:
        c_score+=5
    batsman_ppg[batsman_id]["batting_score"]=c_score

#Step 8: Evaluate bowler performance
def evaluate_bowler(bowlerid_to_details,bowler_ppg,bowlerid):
    if bowlerid not in bowler_ppg:
        bowler_ppg[bowlerid]=dict()
    for matchid in bowlerid_to_details[bowlerid]["matches"]:
        score=0
        w4=bowlerid_to_details[bowlerid]["matches"][matchid]["4w"]
        av=bowlerid_to_details[bowlerid]["matches"][matchid]["runs"]/max(1,bowlerid_to_details[bowlerid]["matches"][matchid]["wickets"])
        sr=bowlerid_to_details[bowlerid]["matches"][matchid]["balls"]/max(1,bowlerid_to_details[bowlerid]["matches"][matchid]["wickets"])
        ec=bowlerid_to_details[bowlerid]["matches"][matchid]["runs"]/max(1,bowlerid_to_details[bowlerid]["matches"][matchid]["balls"]//6)
        dt=bowlerid_to_details[bowlerid]["matches"][matchid]["date"]
        if sr<=15:
            score+=30
        elif sr<=19:
            score+=20
        elif sr<=24:
            score+=10
        if ec<=3:
            score+=50
        elif ec<=5:
            score+=40
        elif ec<7:
            score+=30
        if av<=20:
            score+=30
        elif av<=30:
            score+=20
        elif av<=40:
            score+=10
        if w4>=4:
            score+=30
        elif w4>=2:
            score+=20
        elif w4==1:
            score+=10
        if matchid not in bowler_ppg[bowlerid]:
            bowler_ppg[bowlerid][matchid]=[dt,score]
        bowler_ppg[bowlerid][matchid][1]+=score
    c_score=0
    g_4w=bowlerid_to_details[bowlerid]["4w"]
    g_av=bowlerid_to_details[bowlerid]["runs"]/max(1,bowlerid_to_details[bowlerid]["wickets"])
    g_sr=bowlerid_to_details[bowlerid]["balls"]/max(1,bowlerid_to_details[bowlerid]["wickets"])
    g_ec=bowlerid_to_details[bowlerid]["runs"]/max(1,bowlerid_to_details[bowlerid]["balls"]//6)
    if g_sr<=15:
        c_score+=30
    elif g_sr<=19:
        c_score+=20
    elif g_sr<=24:
        c_score+=10
    if g_ec<=3:
        c_score+=50
    elif g_ec<=5:
        c_score+=40
    elif g_ec<=7:
        c_score+=30
    if g_av<=20:
        c_score+=30
    elif g_av<=30:
        c_score+=20
    elif g_av<=40:
        c_score+=10
    if g_4w>=4:
        c_score+=30
    elif g_4w>=2:
        c_score+=20
    elif g_4w==1:
        c_score+=10
    bowler_ppg[bowlerid]["bowling_score"]=c_score

#Apply criteria for eligible batsman
batsman_ppg=dict()
for batsman_id in batsmanid_to_details:
    if batsmanid_to_details[batsman_id]["runs"]>100:
        evaluate_batsman(batsmanid_to_details=batsmanid_to_details,batsman_ppg=batsman_ppg,batsman_id=batsman_id)

#Apply criteria for eligible bowler
bowler_ppg=dict()
for bowlerid in bowlerid_to_details:
    if bowlerid_to_details[bowlerid]["wickets"]>10:
        evaluate_bowler(bowlerid_to_details=bowlerid_to_details,bowler_ppg=bowler_ppg,bowlerid=bowlerid)

#Apply criteria for eligible allrounder
all_rounder_ppg=dict()
batsmen=set(batsmanid_to_details.keys())
bowlers=set(bowlerid_to_details.keys())
all_rounders=batsmen.intersection(bowlers)
for i in all_rounders:
    if batsmanid_to_details[i]["runs"]>10 and bowlerid_to_details[i]["wickets"]>=1 and len(bowlerid_to_details[i]["matches"])>=2:
        evaluate_batsman(batsmanid_to_details,all_rounder_ppg,i)
        evaluate_bowler(bowlerid_to_details,all_rounder_ppg,i)

#Apply criteria for eligible wicketkeeper
wicket_keeper_ppg=dict()
for batsman_id in batsmanid_to_details:
    if batsmanid_to_details[batsman_id]["wicket_keeper"]>=2:
        evaluate_batsman(batsmanid_to_details=batsmanid_to_details,batsman_ppg=wicket_keeper_ppg,batsman_id=batsman_id)

#Determine the maximum metrics using Recency, Consistency and Form factors

max_d=dict()
max_d["batsman"]=[0,0,0]
max_d["bowler"]=[0,0,0]
max_d["all_rounder"]=[0,0,0]
for i in batsman_ppg:
    r,c=get_form_factors(batsman_ppg,i)
    max_d["batsman"][0]=max(max_d["batsman"][0],r)
    max_d["batsman"][1]=max(max_d["batsman"][1],c)
    max_d["batsman"][2]=max(max_d["batsman"][2],batsman_ppg[i]["batting_score"])
for i in bowler_ppg:
    r,c=get_form_factors(bowler_ppg,i)
    max_d["bowler"][0]=max(max_d["bowler"][0],r)
    max_d["bowler"][1]=max(max_d["bowler"][1],c)
    max_d["bowler"][2]=max(max_d["bowler"][2],bowler_ppg[i]["bowling_score"])
for i in all_rounder_ppg:
    r,c=get_form_factors(all_rounder_ppg,i)
    max_d["all_rounder"][0]=max(max_d["all_rounder"][0],r)
    max_d["all_rounder"][1]=max(max_d["all_rounder"][1],c)
    max_d["all_rounder"][2]=max(max_d["all_rounder"][2],all_rounder_ppg[i]["bowling_score"]+all_rounder_ppg[i]["batting_score"])

#Rank players as per their metrics

bowler_rankings=get_rankings(bowler_ppg,max_d=max_d,type="bowler")
batsman_rankings=get_rankings(batsman_ppg,max_d=max_d,type="batsman")
all_rounder_rankings=get_rankings(all_rounder_ppg,max_d=max_d,type="all_rounder")
wicket_keeper_rankings=get_rankings(wicket_keeper_ppg,max_d=max_d,type="batsman")

#Plot a radar plot based on player's performance

def radar_plot(player_id,batsman_ppg,bowler_ppg,path,max_d):
    attributes=["batting_recency","batting_consistency","batting_score","bowling_consistency","bowling_recency","bowling_score"]
    values=[0,0,0,0,0,0]
    if player_id in batsman_ppg:
        values[0],values[1]=get_form_factors(batsman_ppg,player_id)
        values[2]=batsman_ppg[player_id]["batting_score"]
        values[0]/=max_d["batsman"][0]*0.01
        values[1]/=max_d["batsman"][1]*0.01
        values[2]/=max_d["batsman"][2]*0.01
    if player_id in bowler_ppg:
        values[3],values[4]=get_form_factors(bowler_ppg,player_id)
        values[5]=bowler_ppg[player_id]["bowling_score"]
        values[3]/=max_d["bowler"][0]*0.01
        values[4]/=max_d["bowler"][1]*0.01
        values[5]/=max_d["bowler"][2]*0.01
    # values[6]=values[1]*10+values[4]*10+values[0]*20+values[3]*20+values[2]*70+values[5]*70
    n=6
    angles=np.linspace(0,2*np.pi,n,endpoint=False).tolist()
    values+=values[:1]
    angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True))
    ax.fill(angles,values,color="blue",alpha=0.25)
    ax.plot(angles,values,color="blue",linewidth=0.5,marker='o')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    for i in range(n):
        if values[i] != 0:
            ax.text(angles[i], values[i] + 0.05, f'{values[i]*100:.2f}', horizontalalignment='right', size=10, color='black', weight='normal')
    ax.set_title("Player Features")
    plt.savefig(path,dpi=600)
    plt.close('all')

#Print top 10 batsmen, bowler, all rounder and wicketkeeper as per ranking

print("Top 10 batsmen")
for i in batsman_rankings[:10]:
    print(batsmanid_to_details[i[1]]["name"],"\t\t",i[1],"\t",i[0])
print("\nTop 10 bowlers")
for i in bowler_rankings[:10]:
    print(bowlerid_to_details[i[1]]["name"],"\t\t",i[1],"\t",i[0])
print("\nTop 10 all rounders")
for i in all_rounder_rankings[:10]:
    print(batsmanid_to_details[i[1]]["name"],"\t\t",i[1],"\t",i[0])
print("\nTop 10 wicket keepers")
for i in wicket_keeper_rankings[:10]:
    print(batsmanid_to_details[i[1]]["name"],"\t\t",i[1],"\t",i[0])
