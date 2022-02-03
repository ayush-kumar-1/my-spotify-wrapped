# %%
import pandas as pd 
import json
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from pywaffle import Waffle

# %%
def streaming_data() -> pd.DataFrame:
    df = pd.read_json("../data/StreamingHistory0.json").append(pd.read_json("../data/StreamingHistory1.json"))
    df['endTime'] = pd.to_datetime(df['endTime'])
    df = df[df.endTime >= pd.to_datetime("01-01-2021")]
    return df.reset_index(drop=True)


# %%
def time_played_by_day(df): 
    #add in time interval of single day
    df["time_interval"] = df.endTime.dt.floor('d')
    time_played_by_day = df.groupby("time_interval").sum().reset_index().sort_values(by="msPlayed",ascending=False)
    time_played_by_day["minutes_played"] = (time_played_by_day.msPlayed/1000/60).astype('int')
    
    #add in month of the year
    df["month"] = df.endTime.dt.month
    time_played_by_month = df.groupby("month").sum()
    time_played_by_month["minutes_played"] = (time_played_by_month.msPlayed/1000/60).astype('int')

    # Aggregate based on day of the year
    time_played_by_day = df.groupby("time_interval").agg({"msPlayed": "sum", 
                                                          "month": "mean"}).reset_index()

    #rescale so graph is across entire year not by month
    time_played_by_day['scaled_ms'] = time_played_by_day['msPlayed']/time_played_by_day['msPlayed'].max()

    #fill in missing days with 0 time listened 
    missing_dates = pd.date_range(start="2021-01-01", end="2021-12-31").difference(time_played_by_day.time_interval)
    missing_df = pd.DataFrame({"time_interval": missing_dates, 
                   "msPlayed": [0 for _ in range(len(missing_dates))], 
                   "scaled_ms": [0 for _ in range(len(missing_dates))]})
    missing_df["month"] = missing_df.time_interval.dt.month 
    time_played_by_day = time_played_by_day.append(missing_df).reset_index(drop=True)
    time_palyed_by_day = time_played_by_day.sort_values("time_interval")
    return time_played_by_day

def total_minutes_listened_by_month(time_played_by_day):
    total_listened = []
    for month in range(1, 13): 
        month_df = time_played_by_day[time_played_by_day.month == month]
        total_listened.append(int(month_df.msPlayed.sum()/1000/60))
    
    return total_listened

def create_12_months_of_music(df, total_minutes_listened, save=False):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", 
              "October", "November", "December"]

    plt.rcParams["figure.figsize"] = [25, 7]
    plt.rcParams["figure.dpi"] = 300

    font = FontProperties()
    font.set_name("Verdana")
    
    monospace = FontProperties()
    monospace.set_name("Consolas")
    
    #setting plot colors by month
    colors_by_month = {
        "January": "#87b5ff", 
        "February": "#ff21e1", 
        "March": "#3db800", 
        "April": "#1a4f00", 
        "May": "#34fa5c", 
        "June": "#bab400",
        "July": "#ffe600", 
        "August": "#7cc400",
        "September": "#c45800", 
        "October": "#ff0000",
        "November": "#0069c4", 
        "December": "#80f6ff"
    }

    weeks_in_month = [5, 4, 5, 4, 4, 4, 5, 4, 4, 5, 4, 4]
    colors = []
    for i, (month, color) in enumerate(colors_by_month.items()): 
        colors += [color for _ in range(weeks_in_month[i])]

    values = (df.sort_values("time_interval").msPlayed/1000/60).astype('int')
    weekly_values = []
    for i in range(52): 
        start_index = i*7 
        end_index = start_index + 7
        weekly_values.append(values[start_index:end_index].sum())

    weekly_values = [int(np.ceil(value/100)) for value in weekly_values]
    fig = plt.figure(
        FigureClass = Waffle, 
        rows = max(weekly_values),
        vertical = False, 
        values = weekly_values, 
        colors = colors, #["green" for _ in range(len(weekly_values))], 
        block_arranging_style="new-line", 
        icons="music", 
        interval_ratio_y = -0.08,
        icon_size = 20,
        tight=True
        )

    fig.text(0.001,0.87, "12 Months of Music", fontproperties = font, fontsize=30) # Title
    fig.text(0.001,0.80, "50,000 Minutes of Listening Across 52 Weeks", fontproperties = font, fontsize=25, alpha=0.8) # Subtitle
    fig.text(0.001,0.75, "One Note = 100 Minutes of Music Listened", fontproperties = monospace, fontsize=20, color="green")

    month_xpos = [0.006, 0.086, 0.15, 0.23, 0.291, 0.356, 0.418, 0.497, 0.56, 0.625, 0.703, 0.767]
    i = 0
    for xpos, month in zip(month_xpos, months): 
        fig.text(xpos, -0.01, month, fontproperties =  monospace, fontsize=18, ha="left", va="center")

    for i, (month, xpos) in enumerate(zip(months, month_xpos)):
        ypos = -0.1
        fig.text(xpos, ypos, f"{total_minutes_listened[i]:,}\n Minutes", fontproperties = monospace, fontsize = 18, 
                 rotation="horizontal",color = colors_by_month[month])    
    
    if save:
        plt.savefig("../plots/12MonthsofMusic.png", bbox_inches = 'tight')  
    else: 
        plt.show()


# %%
def create_yearly_plot(df, save = False, 
                       title="365 Days of Spotify", subtitle = "My Listening Throughout the Year", 
                       colormap = plt.cm.Greens):
    """
    Creates a heatmap of listening activity by month. 
    
    Params: 
    df: pd.DataFrame with three columns msPlayed, date of the day listened, and month listened as an integer
    """
    time_played_by_day = df
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", 
              "October", "November", "December"]
    
    #because spotify
    plt.rcParams["figure.figsize"] = [25, 10]
    plt.rcParams["figure.dpi"] = 300

    #font settings 
    font = FontProperties()
    font.set_name("Verdana")

    fig, axes = plt.subplots(12, 1)

    for i, ax in enumerate(axes): 

        month = time_played_by_day[time_played_by_day.month == i + 1].sort_values("time_interval")
        waffle = np.array([month.msPlayed])

        im = ax.matshow(waffle,cmap=colormap)
        if i == 0: 
            ax.set_xticks(ticks = [_ for _ in range(31)], labels = [_ for _ in range(1, 32)], 
                          fontproperties = font, fontsize=20, minor=False)
        else: 
            ax.set_xticks([])
        ax.set_xticks(ticks = [0.5 + i for i in range(31)], color = 'w', minor='True')
        ax.set_yticks([])
        ax.grid(color='w', linestyle='-', linewidth=2, which="minor")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.tick_params(axis = 'both', color="white")
        ax.tick_params(axis = 'both', color="white", which='minor')

        ax.text(-0.65, 0.15, months[i], fontproperties = font, fontsize=20, ha="right")

    axes[0].text(-3, -2.35, title, fontproperties = font,  fontsize=46)
    axes[0].text(-3, -1.5, "My Listening Throughout The Year", fontproperties = font, fontsize=30, alpha = 0.8)

    #adjust spacing between subplots
    spacing = 1.5
    plt.subplots_adjust(top=spacing)
    
    if save: 
        plt.savefig(save, bbox_inches = 'tight', transparent=False)
    else: 
        plt.show()

# %%
def remove_podcasts(df): 
    podcasts = ["Dimension 20", "Ologies with Alie Ward", "The Yard", "Dear Hank & John", "You Can Sit With Us", 
                "The TryPod", "Cortex", "Erin is the Funny One", "The Gus & Eddy Podcast", 'A Conversation With...', 
                "The Numberphile Podcast", "The Joe Rogan Experience", "Lex Fridman Podcast", "Good For You"]
    songs = df[~df.artistName.isin(podcasts)]
    return songs

def find_top_n_songs(songs, n=10):
    podcasts = ["Dimension 20", "Ologies with Alie Ward", "The Yard", "Dear Hank & John", "You Can Sit With Us", 
                "The TryPod", "Cortex", "Erin is the Funny One", "The Gus & Eddy Podcast", 'A Conversation With...', 
                "The Numberphile Podcast", "The Joe Rogan Experience", "Lex Fridman Podcast", "Good For You"]
    
    top_artists = df.groupby("artistName").sum().sort_values(by='msPlayed', ascending=False).reset_index()
    top_ten = top_artists[~top_artists.artistName.isin(podcasts)][:10]
    top_ten["minutes_played"] = (top_ten.msPlayed/1000/60).astype('int')
    top_ten.reset_index(drop=True)
    return top_ten
    
def plot_top_ten_artists(top_ten, scale = 4, save=False): 
    scale = 2
    plt.rcParams["figure.figsize"] = [10/scale, 14/scale]
    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots()
    xs = np.arange(1,11)
    ys = np.sqrt(top_ten.minutes_played)

    font = FontProperties()
    font.set_name("Verdana")

    colors = "#8cff90"
    text_color = 'black'
    background_color = "white"

    ax.barh(xs, ys, joinstyle='miter', color=colors)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    labels = list(top_ten.artistName)
    labels[8] = "Stop Light\nObservations" 

    i = 1
    for label, x, y in zip(labels, xs, top_ten.minutes_played):
        x_loc = 1
        text = f"#{i}. {label}"
        i+= 1
        if label == "Taylor Swift": 
            ax.text(x_loc, x, text, va="center", ha = 'left', fontproperties=font, color = text_color)    
            ax.text(np.sqrt(y) - 1, x, f"2,512 Minutes",va="center", ha="right", fontproperties=font, color=text_color)
        else: 
            ax.text(x_loc, x, text, va="center", ha = 'left', ma = 'right', fontproperties=font)    
            ax.text(np.sqrt(y) + 0.5, x, f"{y} Minutes",va="center", fontproperties=font, color = text_color)

    ## make bars rounded
    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="round,pad=-0.0040,rounding_size=2",
                                ec="none", fc=color,
                                mutation_aspect=0.2
                                )
        patch.remove()
        new_patches.append(p_bbox)

    for patch in new_patches:
        ax.add_patch(patch)

    #change background color
    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    #title 
    ax.text(np.sqrt(2512), 0.3, "My Top Artists 2021", fontproperties = font, fontsize = 20, ha = "right")
    
    if save: 
        plt.savefig(save)
    else:
        plt.show()
        
