import os
import shutil

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from yaml_utils import *


result_yamls_path = os.path.join(".", "result_yamls")
imgs_path = os.path.join(".", "imgs")

def draw_plots(
    figsize = (8, 5), 

    x_list = [], 
    y_list_list = [], 
    y_label_list = [], 
    marker_list = ['o', 's', '*', 'd', '+'], 

    plot_title = None, 
    plot_x_label = None,
    plot_y_label = None, 

    show_grid = True, 
    show_legend = True, 
):

    fig, ax = plt.subplots(
        figsize = figsize
    )

    for i, y_list in enumerate(y_list_list):
        ax.plot(
            x_list, 
            y_list, 
            marker = marker_list[i % len(marker_list)], 
            label = y_label_list[i]
        )

    if plot_title is not None:
        ax.set_title(plot_title)
    if plot_x_label is not None:
        ax.set_xlabel(plot_x_label)
    if plot_y_label is not None:
        ax.set_ylabel(plot_y_label)
    
    ax.grid(show_grid)
    if show_legend:
        ax.legend()

    return fig, ax

def merge_plot_list(
    plot_list = [], 

    figsize = (10, 5), 
    num_rows = 1, num_cols = 1, 

    show_grid_list = [], 
    show_legend_list = [], 
):
    
    new_fig, new_ax_list = plt.subplots(
        num_rows, num_cols, 
        figsize = figsize
    )

    for i, (_, ax) in enumerate(plot_list):
        for line in ax.get_lines():
            new_ax_list[i].plot(
                line.get_xdata(), line.get_ydata(), 
                marker = line.get_marker(), 
                color = line.get_color(), 
                label = line.get_label()
            )
        
        new_ax_list[i].set_title(ax.get_title())
        new_ax_list[i].set_xlabel(ax.get_xlabel())
        new_ax_list[i].set_ylabel(ax.get_ylabel())
 
        new_ax_list[i].grid(show_grid_list[i])
        if show_legend_list[i]:
            new_ax_list[i].legend()

    # adjust interval between subplots
    plt.tight_layout()

    return new_fig

def get_success_rate_lists(
    yaml_path, 
    open_dof_list_key
):

    yaml_object = load_yaml(yaml_path)

    open_dof_list = eval(
        get_yaml_item(
            yaml_object, 
            key = open_dof_list_key
        )
    )

    RGBManip_train_success_rate_list = eval(
        get_yaml_item(
            yaml_object, 
            key = "RGBManip_train_success_rate_list"
        )
    )
    RGBManip_test_success_rate_list = eval(
        get_yaml_item(
            yaml_object, 
            key = "RGBManip_test_success_rate_list"
        )
    )

    Ours_train_success_rate_list = eval(
        get_yaml_item(
            yaml_object, 
            key = "Ours_train_success_rate_list"
        )
    )
    Ours_test_success_rate_list = eval(
        get_yaml_item(
            yaml_object, 
            key = "Ours_test_success_rate_list"
        )
    )

    return open_dof_list, \
        RGBManip_train_success_rate_list, RGBManip_test_success_rate_list, \
        Ours_train_success_rate_list, Ours_test_success_rate_list

def draw_open_door():

    yaml_path = os.path.join(result_yamls_path, "open_door_success_rate.yaml")

    open_door_dof_list, \
        RGBManip_train_success_rate_list, RGBManip_test_success_rate_list, \
        Ours_train_success_rate_list, Ours_test_success_rate_list \
            = get_success_rate_lists(
                yaml_path, 
                open_dof_list_key = "open_door_dof_list"
            )

    train_fig, train_ax = draw_plots(
        figsize = (6, 5), 

        x_list = open_door_dof_list, 
        y_list_list = [
            RGBManip_train_success_rate_list, 
            Ours_train_success_rate_list
        ], 
        y_label_list = [
            "RGBManip", 
            "Ours"
        ], 

        plot_title = "Open Door Train", 
        plot_x_label = "Angle (degrees)", 
        plot_y_label = "Success Rate (%)", 

        show_grid = True, 
        show_legend = True
    )

    test_fig, test_ax = draw_plots(
        figsize = (6, 5), 

        x_list = open_door_dof_list, 
        y_list_list = [
            RGBManip_test_success_rate_list, 
            Ours_test_success_rate_list 
        ], 
        y_label_list = [
            "RGBManip", 
            "Ours"
        ], 

        plot_title = "Open Door Test", 
        plot_x_label = "Angle (degrees)", 
        plot_y_label = "Success Rate (%)", 

        show_grid = True, 
        show_legend = True
    )

    merged_fig = merge_plot_list(
        plot_list = [
            (train_fig, train_ax), 
            (test_fig, test_ax)
        ], 

        figsize = (12, 5), 
        num_rows = 1, num_cols = 2, 

        show_grid_list = [True, True], 
        show_legend_list = [True, True], 
    )

    merged_fig_name = "open_door_merged_success_rate.png"
    merged_fig.savefig(
        os.path.join(imgs_path, merged_fig_name)
    )

def draw_open_drawer():

    yaml_path = os.path.join(result_yamls_path, "open_drawer_success_rate.yaml")

    open_drawer_dof_list, \
        RGBManip_train_success_rate_list, RGBManip_test_success_rate_list, \
        Ours_train_success_rate_list, Ours_test_success_rate_list \
            = get_success_rate_lists(
                yaml_path, 
                open_dof_list_key = "open_drawer_dof_list"
            )

    train_fig, train_ax = draw_plots(
        figsize = (6, 5), 

        x_list = open_drawer_dof_list, 
        y_list_list = [
            RGBManip_train_success_rate_list, 
            Ours_train_success_rate_list
        ], 
        y_label_list = [
            "RGBManip", 
            "Ours"
        ], 

        plot_title = "Open Drawer Train", 
        plot_x_label = "Distance (cm)", 
        plot_y_label = "Success Rate (%)", 

        show_grid = True, 
        show_legend = True
    )

    test_fig, test_ax = draw_plots(
        figsize = (6, 5), 

        x_list = open_drawer_dof_list, 
        y_list_list = [
            RGBManip_test_success_rate_list, 
            Ours_test_success_rate_list 
        ], 
        y_label_list = [
            "RGBManip", 
            "Ours"
        ], 

        plot_title = "Open Drawer Test", 
        plot_x_label = "Distance (cm)", 
        plot_y_label = "Success Rate (%)", 

        show_grid = True, 
        show_legend = True
    )

    merged_fig = merge_plot_list(
        plot_list = [
            (train_fig, train_ax), 
            (test_fig, test_ax)
        ], 

        figsize = (12, 5), 
        num_rows = 1, num_cols = 2, 

        show_grid_list = [True, True], 
        show_legend_list = [True, True], 
    )

    merged_fig_name = "open_drawer_merged_success_rate.png"
    merged_fig.savefig(
        os.path.join(imgs_path, merged_fig_name)
    )


if __name__ == "__main__":

    draw_open_drawer()

    draw_open_door()
