def makeMatchSummary(home_team_name,  # string (should agree with understat naming)
                     away_team_name,  # string
                     ustatxg,  # tuple (home_xg, away_xg)
                     f38xg,  # tuple
                     infogolxg,  # tuple
                     home_odds,  # tuple (odds of 50/23 input as (50, 23))
                     draw_odds,  # tuple
                     away_odds,  # tuple
                     f38nsxg,  # tuple in (home_nsxg, away_nsxg)
                     ustatid,  # match id from understat (the number at the end of the url for the game)
                     correct_score_odds_csv_path,  # path to csv file containing correct score odds from oddsportal
                     home_colour='darkblue',
                     away_colour='darkorange'):
    # To make the oddsportal csv, just copy & paste the correct score odds table from oddsportal into an excel/google sheet.
    # Add a '(' to the entry in the final cell to change it from, say, '0:9150/1' to '0:9150/1('
    # The pre-match odds can be found on the match's oddsportal page
    # (e.g. https://www.oddsportal.com/soccer/england/premier-league/chelsea-manchester-united-O6chyK0Q/#cs;2)

    import shutil
    import pandas as pd
    import matplotlib.pyplot as plt
    from poibin.poibin import PoiBin
    import numpy as np
    import matplotlib.colors as mcol
    import asyncio
    import json
    import aiohttp
    from understat import Understat
    from matplotlib import rcParams
    from highlight_text.htext import fig_htext
    import imageio
    import matplotlib as mpl
    import os

    correct_score_odds = pd.read_csv(correct_score_odds_csv_path,
                                     header=None)
    ho_s = np.empty(len(correct_score_odds))
    aw_s = np.empty(len(correct_score_odds))
    odds_string = np.empty(len(correct_score_odds))
    for k in range(len(correct_score_odds)):
        home_score, rest = correct_score_odds[0][k].split(':')
        ho_s[k] = home_score
        if rest[1] == '0':
            aw_s[k] = rest[0] + rest[1]
            rest = rest[2:]
        else:
            aw_s[k] = rest[0]
            rest = rest[1:]
        odds, rest = rest.split('(')
        c = odds.split('/')
        odds_string[k] = float(c[1]) / (float(c[0]) + float(c[1]))

    home_score_pre_match = np.sum(ho_s * odds_string)
    away_score_pre_match = np.sum(aw_s * odds_string)

    home_win_prob = home_odds[1] / (home_odds[0] + home_odds[1])
    away_win_prob = away_odds[1] / (away_odds[0] + away_odds[1])
    draw_prob = draw_odds[1] / (draw_odds[0] + draw_odds[1])

    pre_match_probs = np.array([home_win_prob, draw_prob, away_win_prob])
    pre_match_probs = pre_match_probs / np.sum(pre_match_probs)

    res_h = []

    async def main():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            teams = await understat.get_teams(
                "epl",
                2019,
                title=home_team_name
            )
            res_h.append(teams)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    ppda_h = [kk['ppda']['att'] / kk['ppda']['def'] for kk in res_h[0][0]['history']]
    ppda_h_allowed = [kk['ppda_allowed']['att'] / kk['ppda_allowed']['def'] for kk in res_h[0][0]['history']]
    deep_h = [kk['deep'] for kk in res_h[0][0]['history']]
    deep_h_allowed = [kk['deep_allowed'] for kk in res_h[0][0]['history']]
    passes_h = [kk['ppda_allowed']['att'] for kk in res_h[0][0]['history']]
    passes_h_allowed = [kk['ppda']['att'] for kk in res_h[0][0]['history']]

    res_a = []

    async def main():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            teams = await understat.get_teams(
                "epl",
                2019,
                title=away_team_name
            )
            res_a.append(teams)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    ppda_a = [kk['ppda']['att'] / kk['ppda']['def'] for kk in res_a[0][0]['history']]
    ppda_a_allowed = [kk['ppda_allowed']['att'] / kk['ppda_allowed']['def'] for kk in res_a[0][0]['history']]
    deep_a = [kk['deep'] for kk in res_a[0][0]['history']]
    deep_a_allowed = [kk['deep_allowed'] for kk in res_a[0][0]['history']]
    passes_a = [kk['ppda_allowed']['att'] for kk in res_a[0][0]['history']]
    passes_a_allowed = [kk['ppda']['att'] for kk in res_a[0][0]['history']]

    res_shots = []

    async def main():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            players = await understat.get_match_shots(ustatid)
            res_shots.append(players)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    ph = np.array([float(kk['xG']) for kk in res_shots[0]['h']])
    pa = np.array([float(kk['xG']) for kk in res_shots[0]['a']])
    reb_inds_h = np.where(np.array([(kk['lastAction']) for kk in res_shots[0]['h']]) == 'Rebound')[0]
    if reb_inds_h.shape[0] > 0:
        ph[reb_inds_h] = ph[reb_inds_h] * (1 - ph[reb_inds_h - 1])
    reb_inds_a = np.where(np.array([(kk['lastAction']) for kk in res_shots[0]['a']]) == 'Rebound')[0]
    if reb_inds_a.shape[0] > 0:
        pa[reb_inds_a] = pa[reb_inds_a] * (1 - pa[reb_inds_a - 1])
    ph = PoiBin(ph)
    pa = PoiBin(pa)
    pa = pa.pmf([k for k in range(np.minimum(15, pa.number_trials))])
    ph = ph.pmf([k for k in range(np.minimum(15, ph.number_trials))])
    mypmf = np.outer(ph, pa)
    xg_probs = np.array([np.sum(np.tril(mypmf, -1)), np.sum(np.diag(mypmf)), np.sum(np.triu(mypmf, 1))])
    xg_probs = xg_probs / np.sum(xg_probs)

    ensemble_xg = (np.array(ustatxg) + np.array(f38xg) + np.array(infogolxg)) / 3.0

    ##########

    rcParams['font.sans-serif'] = "Palatino Linotype"
    rcParams['font.family'] = "sans-serif"

    ymax = 100 / 68
    xmax = 1
    xh = np.array([float(kk['Y']) for kk in res_shots[0]['h']])
    xh = xh * xmax
    yh = np.array([float(kk['X']) for kk in res_shots[0]['h']])
    yh = yh * ymax
    xgh = np.array([float(kk['xG']) for kk in res_shots[0]['h']])
    minute_h = np.array([float(kk['minute']) for kk in res_shots[0]['h']])
    result_h = np.array([kk['result'] for kk in res_shots[0]['h']])
    xa = np.array([float(kk['Y']) for kk in res_shots[0]['a']])
    xa = xa * xmax
    ya = np.array([float(kk['X']) for kk in res_shots[0]['a']])
    ya = ya * ymax
    xga = np.array([float(kk['xG']) for kk in res_shots[0]['a']])
    minute_a = np.array([float(kk['minute']) for kk in res_shots[0]['a']])
    result_a = np.array([kk['result'] for kk in res_shots[0]['a']])

    alpha_h = np.zeros_like(xgh)
    alpha_a = np.zeros_like(xga)
    used_h = np.zeros_like(xgh)
    used_a = np.zeros_like(xga)

    h_cols = np.tile(mcol.to_rgb(home_colour), (xgh.shape[0], 1))
    a_cols = np.tile(mcol.to_rgb(away_colour), (xga.shape[0], 1))

    mpl.rcParams['lines.linewidth'] = 0.3
    mpl.rcParams['patch.linewidth'] = 0.3

    plt.figure(dpi=300, figsize=(7.2, 12.8))
    plt.axis('off')
    ax = plt.gca()

    # draw pitch

    ax.add_patch(plt.Rectangle((0, ymax / 2), xmax, ymax / 2, fill=False))
    ax.plot([0, xmax],
            [ymax / 2, ymax / 2],
            c='lightgray', zorder=-10)
    ax.plot(xmax / 2 + 9.15 * ymax / 100 * np.cos(np.linspace(0, np.pi, 100)),
            ymax / 2 + 9.15 * ymax / 100 * np.sin(np.linspace(0, np.pi, 100)),
            c='lightgray', zorder=-10)
    ax.plot([xmax / 2 - 20.15 * ymax / 100, xmax / 2 - 20.15 * ymax / 100],
            [ymax, ymax - ymax / 100 * 16.5],
            c='lightgray', zorder=-10)
    ax.plot([xmax / 2 + 20.15 * ymax / 100, xmax / 2 + 20.15 * ymax / 100],
            [ymax, ymax - ymax / 100 * 16.5],
            c='lightgray', zorder=-10)
    ax.plot([xmax / 2 - 20.15 * ymax / 100, xmax / 2 + 20.15 * ymax / 100],
            [ymax - ymax / 100 * 16.5, ymax - ymax / 100 * 16.5],
            c='lightgray', zorder=-10)
    ax.plot(xmax / 2 + ymax / 100 * 9.15 * np.cos(
        np.linspace(-np.pi / 2 + np.arccos(5.5 / 9.15),
                    -np.pi / 2 - np.arccos(5.5 / 9.15),
                    100)),
            ymax - ymax / 100 * 11 + ymax / 100 * 9.15 * np.sin(
                np.linspace(-np.pi / 2 + np.arccos(5.5 / 9.15),
                            -np.pi / 2 - np.arccos(5.5 / 9.15),
                            100)),
            c='lightgray', zorder=-10)

    ## top panels (pre-match odds, xg total, nsxg vs xg plot)

    # nsxg vs xg

    plt.arrow(0.28, ymax + 0.1, 0, 0.4, head_width=0.01, head_length=0.02, zorder=-13, linewidth=0.2,
              length_includes_head=True, fc='black', ec='black')
    ax.text(0.28, ymax + 0.3, 'Threat', rotation=90, ha='center', va='center', fontsize=3,
            bbox=dict(facecolor='white', alpha=1, edgecolor='white', pad=0), zorder=-10)

    plt.arrow(0.3, ymax + 0.07, 0.4, 0, head_width=0.02, head_length=0.02, zorder=-13, linewidth=0.2,
              length_includes_head=True, fc='black', ec='black')
    ax.text(0.5, ymax + 0.07, 'Efficiency', ha='center', va='center', fontsize=3,
            bbox=dict(facecolor='white', alpha=1, edgecolor='white', pad=0), zorder=-10)

    plt.arrow(0.43, ymax + 0.13, -0.1, 0.1, head_width=0.015, head_length=0.015, zorder=-13, linewidth=0.2,
              length_includes_head=True, fc='gray', ec='gray')
    plt.arrow(0.57, ymax + 0.13, 0.1, 0.1, head_width=0.015, head_length=0.015, zorder=-13, linewidth=0.2,
              length_includes_head=True, fc='gray', ec='gray')

    plt.plot([0.5, 0.3], [ymax + 0.1, ymax + 0.3], c='lightgray')
    plt.plot([0.5, 0.7], [ymax + 0.1, ymax + 0.3], c='lightgray')
    plt.plot([0.5, 0.3], [ymax + 0.5, ymax + 0.3], c='lightgray')
    plt.plot([0.5, 0.7], [ymax + 0.5, ymax + 0.3], c='lightgray')
    plt.plot([0.4, 0.6], [ymax + 0.2, ymax + 0.4], c='lightgray', ls='--')
    plt.plot([0.6, 0.4], [ymax + 0.2, ymax + 0.4], c='lightgray', ls='--')

    coords_h1 = np.minimum(ensemble_xg[0], 3) + np.minimum(f38nsxg[0], 3)
    coords_h2 = np.minimum(ensemble_xg[0], 3) - np.minimum(f38nsxg[0], 3)
    coords_a1 = np.minimum(ensemble_xg[1], 3) + np.minimum(f38nsxg[1], 3)
    coords_a2 = np.minimum(ensemble_xg[1], 3) - np.minimum(f38nsxg[1], 3)

    hxhx = 0.3 + (3 + coords_h2) / 6 * 0.4
    axax = 0.3 + (3 + coords_a2) / 6 * 0.4

    hyhy = ymax + 0.1 + coords_h1 / 6 * 0.4
    ayay = ymax + 0.1 + coords_a1 / 6 * 0.4

    plt.plot(hxhx, hyhy, '.', c=home_colour, ms=5)
    plt.plot(axax, ayay, '.', c=away_colour, ms=5)

    # pre-match predictions

    ax.add_patch(plt.Rectangle((-0.25, ymax + 0.05), 0.5, 0.5, fill=False))
    ax.text(0, ymax + 0.53, 'Pre-match prediction',
            fontsize=3, ha='center', va='top')
    ax.text(0, ymax + 0.07,
            "%.2f" % home_score_pre_match + ' - ' + "%.2f" % away_score_pre_match,
            fontsize=6, ha='center', va='bottom')

    pre_match_widths = 0.3 * pre_match_probs
    pre_match_endpoints = np.cumsum(0.3 * pre_match_probs)

    ax.add_patch(plt.Rectangle((-0.15, ymax + 0.3),
                               pre_match_widths[0],
                               0.1,
                               facecolor="darkblue",
                               alpha=0.3,
                               edgecolor='black'))
    ax.text(-0.15 + 0.005, ymax + 0.43, "%.3g" % (100 * pre_match_probs[0]) + '%',
            fontsize=2.75, ha='left', va='center', c=home_colour)
    ax.add_patch(plt.Rectangle((-0.15 + pre_match_endpoints[0], ymax + 0.3),
                               pre_match_widths[1],
                               0.1,
                               facecolor='darkgray',
                               alpha=0.3,
                               edgecolor='black'))
    ax.text(-0.15 + pre_match_endpoints[0] + 0.005, ymax + 0.26,
            "%.3g" % (100 * pre_match_probs[1]) + '%',
            fontsize=2.75, ha='left', va='center')
    ax.add_patch(plt.Rectangle((-0.15 + pre_match_endpoints[1], ymax + 0.3),
                               pre_match_widths[2],
                               0.1,
                               facecolor=away_colour,
                               alpha=0.3,
                               edgecolor='black'))
    ax.text(-0.15 + pre_match_endpoints[2] - 0.005, ymax + 0.43,
            "%.3g" % (100 * pre_match_probs[2]) + '%',
            fontsize=2.75, ha='right', va='center', c=away_colour)

    # post-match xg

    ax.add_patch(plt.Rectangle((0.75, ymax + 0.05), 0.5, 0.5, fill=False))
    ax.text(1, ymax + 0.53, 'Post-match xG',
            fontsize=3, ha='center', va='top')
    ax.text(1, ymax + 0.07,
            "%.2f" % ensemble_xg[0] + ' - ' + "%.2f" % ensemble_xg[1],
            fontsize=6, ha='center', va='bottom')

    xg_widths = 0.3 * xg_probs
    xg_endpoints = np.cumsum(0.3 * xg_probs)

    ax.add_patch(plt.Rectangle((0.85, ymax + 0.3),
                               xg_widths[0],
                               0.1,
                               facecolor="darkblue",
                               alpha=0.3,
                               edgecolor='black'))
    ax.text(0.85 + 0.005, ymax + 0.43, "%.3g" % (100 * xg_probs[0]) + '%',
            fontsize=2.75, ha='left', va='center', c=home_colour)
    ax.add_patch(plt.Rectangle((0.85 + xg_endpoints[0], ymax + 0.3),
                               xg_widths[1],
                               0.1,
                               facecolor='darkgray',
                               alpha=0.3,
                               edgecolor='black'))
    ax.text(0.85 + xg_endpoints[0] + 0.005, ymax + 0.26,
            "%.3g" % (100 * xg_probs[1]) + '%',
            fontsize=2.75, ha='left', va='center')
    ax.add_patch(plt.Rectangle((0.85 + xg_endpoints[1], ymax + 0.3),
                               xg_widths[2],
                               0.1,
                               facecolor=away_colour,
                               alpha=0.3,
                               edgecolor='black'))
    ax.text(0.85 + xg_endpoints[2] - 0.005, ymax + 0.43,
            "%.3g" % (100 * xg_probs[2]) + '%',
            fontsize=2.75, ha='right', va='center', c=away_colour)

    ## Match stats panel (bottom)

    ax.add_patch(plt.Rectangle((-0.25, ymax / 2 - 0.05 - 1), 1.5, 1, fill=False))

    ax.text(0, ymax / 2 - 0.55, 'PPDA',
            ha='center', va='center', fontsize=5)
    ax.text(0.5, ymax / 2 - 0.55, 'Opp. half passes',
            ha='center', va='center', fontsize=5)
    ax.text(1, ymax / 2 - 0.55, 'Deep completions',
            ha='center', va='center', fontsize=5)

    ax.text(0.25, ymax / 2 - 0.1, "Team's season avg.",
            ha='center', va='center', fontsize=3.5, color='seagreen')
    ax.text(0.5, ymax / 2 - 0.1, "Avg. allowed by today's opp.",
            ha='center', va='center', fontsize=3.5, color='firebrick')
    ax.text(0.75, ymax / 2 - 0.1, 'This game',
            ha='center', va='center', fontsize=3.5, color='darkkhaki')

    ax.add_patch(plt.Rectangle((-0.2, ymax / 2 - 0.05 - 0.8), 0.4, 0.2, fill=False, color=away_colour))
    ax.add_patch(plt.Rectangle((-0.2, ymax / 2 - 0.05 - 0.4), 0.4, 0.2, fill=False, color=home_colour))
    ax.text(-0.2, ymax / 2 - 0.05 - 0.82, '0',
            ha='center', va='top', fontsize=4)
    ax.text(0.2, ymax / 2 - 0.05 - 0.82, '20',
            ha='center', va='top', fontsize=4)

    ppda_this_match_h = np.minimum(ppda_h[-1] / 20, 1)
    ppda_avg_h = np.minimum(np.mean(ppda_h) / 20, 1)
    ppda_avg_h_oppo = np.minimum(np.mean(ppda_a_allowed) / 20, 1)

    ax.add_patch(
        plt.Rectangle((-0.2, ymax / 2 - 0.05 - 0.4), ppda_this_match_h * 0.4, 0.2,
                      color='darkkhaki', zorder=-15, alpha=0.75))
    plt.plot([-0.2 + ppda_avg_h * 0.4, -0.2 + ppda_avg_h * 0.4],
             [ymax / 2 - 0.05 - 0.2, ymax / 2 - 0.05 - 0.4],
             c='seagreen', linewidth=2, alpha=1)
    plt.plot([-0.2 + ppda_avg_h_oppo * 0.4, -0.2 + ppda_avg_h_oppo * 0.4],
             [ymax / 2 - 0.05 - 0.2, ymax / 2 - 0.05 - 0.4],
             c='firebrick', linewidth=2, alpha=1)

    ppda_this_match_a = np.minimum(ppda_a[-1] / 20, 1)
    ppda_avg_a = np.minimum(np.mean(ppda_a) / 20, 1)
    ppda_avg_a_oppo = np.minimum(np.mean(ppda_h_allowed) / 20, 1)

    ax.add_patch(
        plt.Rectangle((-0.2, ymax / 2 - 0.05 - 0.8), ppda_this_match_a * 0.4, 0.2,
                      color='darkkhaki', zorder=-15, alpha=0.75))

    plt.plot([-0.2 + ppda_avg_a * 0.4, -0.2 + ppda_avg_a * 0.4],
             [ymax / 2 - 0.05 - 0.8, ymax / 2 - 0.05 - 0.6],
             c='seagreen', linewidth=2, alpha=1)
    plt.plot([-0.2 + ppda_avg_a_oppo * 0.4, -0.2 + ppda_avg_a_oppo * 0.4],
             [ymax / 2 - 0.05 - 0.8, ymax / 2 - 0.05 - 0.6],
             c='firebrick', linewidth=2, alpha=1)
    #####
    ax.add_patch(plt.Rectangle((0.3, ymax / 2 - 0.05 - 0.8), 0.4, 0.2, fill=False, color=away_colour))
    ax.add_patch(plt.Rectangle((0.3, ymax / 2 - 0.05 - 0.4), 0.4, 0.2, fill=False, color=home_colour))
    ax.text(0.3, ymax / 2 - 0.05 - 0.82, '75', ha='center', va='top', fontsize=4)
    ax.text(0.7, ymax / 2 - 0.05 - 0.82, '350', ha='center', va='top', fontsize=4)

    passes_this_match_h = np.maximum(0, np.minimum((passes_h[-1] - 50) / 275, 1))
    passes_avg_h = np.maximum(0, np.minimum((np.mean(passes_h) - 50) / 275, 1))
    passes_avg_h_oppo = np.maximum(0, np.minimum((np.mean(passes_a_allowed) - 50) / 275, 1))

    ax.add_patch(
        plt.Rectangle((0.3, ymax / 2 - 0.05 - 0.4), passes_this_match_h * 0.4, 0.2,
                      color='darkkhaki', zorder=-15, alpha=0.75))
    plt.plot([0.3 + passes_avg_h * 0.4, 0.3 + passes_avg_h * 0.4],
             [ymax / 2 - 0.05 - 0.2, ymax / 2 - 0.05 - 0.4],
             c='seagreen', linewidth=2, alpha=1)
    plt.plot([0.3 + passes_avg_h_oppo * 0.4, 0.3 + passes_avg_h_oppo * 0.4],
             [ymax / 2 - 0.05 - 0.2, ymax / 2 - 0.05 - 0.4],
             c='firebrick', linewidth=2, alpha=1)

    passes_this_match_a = np.maximum(0, np.minimum((passes_a[-1] - 50) / 275, 1))
    passes_avg_a = np.maximum(0, np.minimum((np.mean(passes_a) - 50) / 275, 1))
    passes_avg_a_oppo = np.maximum(0, np.minimum((np.mean(passes_h_allowed) - 50) / 275, 1))

    ax.add_patch(
        plt.Rectangle((0.3, ymax / 2 - 0.05 - 0.8), passes_this_match_a * 0.4, 0.2,
                      color='darkkhaki', zorder=-15, alpha=0.75))
    plt.plot([0.3 + passes_avg_a * 0.4, 0.3 + passes_avg_a * 0.4],
             [ymax / 2 - 0.05 - 0.8, ymax / 2 - 0.05 - 0.6],
             c='seagreen', linewidth=2, alpha=1)
    plt.plot([0.3 + passes_avg_a_oppo * 0.4, 0.3 + passes_avg_a_oppo * 0.4],
             [ymax / 2 - 0.05 - 0.8, ymax / 2 - 0.05 - 0.6],
             c='firebrick', linewidth=2, alpha=1)
    #####
    ax.add_patch(plt.Rectangle((0.8, ymax / 2 - 0.05 - 0.8), 0.4, 0.2, fill=False, color=away_colour))
    ax.add_patch(plt.Rectangle((0.8, ymax / 2 - 0.05 - 0.4), 0.4, 0.2, fill=False, color=home_colour))
    ax.text(0.8, ymax / 2 - 0.05 - 0.82, '0', ha='center', va='top', fontsize=4)
    ax.text(1.2, ymax / 2 - 0.05 - 0.82, '25', ha='center', va='top', fontsize=4)

    deep_this_match_h = np.maximum(0, np.minimum(deep_h[-1] / 25, 1))
    deep_avg_h = np.maximum(0, np.minimum(np.mean(deep_h) / 25, 1))
    deep_avg_h_oppo = np.maximum(0, np.minimum(np.mean(deep_a_allowed) / 25, 1))

    ax.add_patch(
        plt.Rectangle((0.8, ymax / 2 - 0.05 - 0.4), deep_this_match_h * 0.4, 0.2,
                      color='darkkhaki', zorder=-15, alpha=0.75))
    plt.plot([0.8 + deep_avg_h * 0.4, 0.8 + deep_avg_h * 0.4], [ymax / 2 - 0.05 - 0.2, ymax / 2 - 0.05 - 0.4],
             c='seagreen', linewidth=2, alpha=1)
    plt.plot([0.8 + deep_avg_h_oppo * 0.4, 0.8 + deep_avg_h_oppo * 0.4],
             [ymax / 2 - 0.05 - 0.2, ymax / 2 - 0.05 - 0.4],
             c='firebrick', linewidth=2, alpha=1)

    deep_this_match_a = np.maximum(0, np.minimum(deep_a[-1] / 25, 1))
    deep_avg_a = np.maximum(0, np.minimum(np.mean(deep_a) / 25, 1))
    deep_avg_a_oppo = np.maximum(0, np.minimum(np.mean(deep_h_allowed) / 25, 1))

    ax.add_patch(
        plt.Rectangle((0.8, ymax / 2 - 0.05 - 0.8), deep_this_match_a * 0.4, 0.2,
                      color='darkkhaki', zorder=-15, alpha=0.75))
    plt.plot([0.8 + deep_avg_a * 0.4, 0.8 + deep_avg_a * 0.4],
             [ymax / 2 - 0.05 - 0.8, ymax / 2 - 0.05 - 0.6],
             c='seagreen', linewidth=2, alpha=1)
    plt.plot([0.8 + deep_avg_a_oppo * 0.4, 0.8 + deep_avg_a_oppo * 0.4],
             [ymax / 2 - 0.05 - 0.8, ymax / 2 - 0.05 - 0.6],
             c='firebrick', linewidth=2, alpha=1)

    cur_home_goals = 0
    cur_away_goals = 0
    p1_h_alpha = 0
    p1_a_alpha = 0

    home_goals = np.sum(result_h == 'Goal') + np.sum(result_a == 'OwnGoal')
    away_goals = np.sum(result_a == 'Goal') + np.sum(result_h == 'OwnGoal')

    fig_htext(s='<' + home_team_name + '>' + ' ' +
                str(home_goals) + ' - ' + str(away_goals) + ' ' +
                '<' + away_team_name + '>',
              x=0.5, y=0.95,
              color='k',
              highlight_colors=[home_colour, away_colour],
              fontsize=7,
              ha='center',
              va='center')
    ax.set_xlim(-0.3, 1.3)

    if os.path.exists(os.getcwd() + '\\tmppngs'):
        for filename in os.listdir(os.getcwd() + '\\tmppngs'):
            filepath = os.path.join(os.getcwd() + '\\tmppngs', filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)
    else:
        os.makedirs(os.getcwd() + '\\tmppngs')

    for minute in range(100):
        p1_h_alpha = np.maximum(p1_h_alpha - 0.1, 0)
        p1_a_alpha = np.maximum(p1_a_alpha - 0.1, 0)
        alpha_h = np.maximum(alpha_h - 0.05, used_h * 0.1)
        alpha_a = np.maximum(alpha_a - 0.05, used_a * 0.1)
        hc = np.c_[h_cols, alpha_h]
        ac = np.c_[a_cols, alpha_a]

        match_h = np.where(minute_h == minute)[0]
        match_a = np.where(minute_a == minute)[0]

        tx3 = ax.text(xmax / 100, ymax - 0.05, str(minute) + ':00', fontsize=4.5, ha='left', va='top')
        if minute != 0:
            ax.add_patch(
                plt.Rectangle(((minute - 1)/99, ymax / 2), 1/99, 0.05,
                          color='black', zorder=-15, alpha=0.75,ec=None))

        if (np.sum(result_h[match_h] == 'Goal') + np.sum(result_a[match_a] == 'OwnGoal')) > 0:
            p1_h_alpha = 1

        if (np.sum(result_a[match_a] == 'Goal') + np.sum(result_h[match_h] == 'OwnGoal')) > 0:
            p1_a_alpha = 1

        alpha_h[match_h] = 1
        alpha_a[match_a] = 1
        used_h[match_h] = 1
        used_a[match_a] = 1

        cur_pts1 = plt.scatter(xh, yh, s=100 * xgh, c=hc)
        cur_pts2 = plt.scatter(xa, ya, s=100 * xga, c=ac)
        tx1 = plt.text(xmax / 10, 0.75, '+1', c=home_colour, alpha=p1_h_alpha, ha='left', fontsize=7)
        tx2 = plt.text(xmax - xmax / 10, 0.75, '+1', c=away_colour, alpha=p1_a_alpha, ha='right', fontsize=7)
        plt.pause(0.05)
        plt.show()
        if minute == 0:
            rotn1 = -plt.gca().transData.transform_angles(np.array((45,)),
                                                          np.array([0.62, ymax + 0.18]).reshape((1, 2)))[0]
            ax.text(0.38, ymax + 0.18, 'nsxG', ha='center', va='center', fontsize=2.5, c='gray', rotation=rotn1,
                    rotation_mode='anchor',
                    bbox=dict(facecolor='white', alpha=1, edgecolor='white', pad=0), zorder=-11)

            ax.text(0.62, ymax + 0.18, 'xG', ha='center', va='center', fontsize=2.5, c='gray', rotation=-rotn1,
                    rotation_mode='anchor',
                    bbox=dict(facecolor='white', alpha=1, edgecolor='white', pad=0), zorder=-11)
        plt.savefig(os.getcwd() + '\\tmppngs\\' + str(minute) + '.png')
        if minute != 99:
            cur_pts1.remove()
            cur_pts2.remove()
            tx1.remove()
            tx2.remove()
            tx3.remove()
    images = []
    for m in range(99):
        images.append(imageio.imread(os.getcwd() + '\\tmppngs\\' + str(m) + '.png'))
    imageio.mimsave(os.getcwd() + '\\movie.gif', images, fps=5)
