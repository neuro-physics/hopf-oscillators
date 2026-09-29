import os
import re
import scipy.io
import warnings
import numpy as np
import pyvista as pv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

pv.set_jupyter_backend('static')

def get_node_initials(aal_labels):
    if type(aal_labels) is np.ndarray:
        if aal_labels.ndim==0:
            aal_labels = aal_labels.item()
    if isinstance(aal_labels,str):
        # Find all matches; the second group index [1] contains the actual character
        matches = re.findall(r'[A-Z0-9]|(?<=_)[a-z]', aal_labels)
        return "".join(matches)
    else:
        return [ get_node_initials(a) for a in aal_labels ]

def _make_discrete_cmap(cmap, n, brain_bg_color):
    base_colors                         = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    bbg_color_arr                       = np.ones(base_colors.shape[1])
    bbg_color_arr[:len(brain_bg_color)] = brain_bg_color
    colors                              = np.vstack( ( bbg_color_arr, base_colors ) )
    return mcolors.ListedColormap(colors)

def _get_surf_AAL_LR_vertices(aal_surf):
    # -------------------------
    # Unwrap MATLAB surf struct
    # -------------------------
    tri   = aal_surf['tri'].item().astype(np.int64) - 1 # -1 for 0-based index
    coord = aal_surf['coord'].item().T

    # Hemisphere split (MATLAB convention)
    n_vertex = coord.shape[0]
    n_tri    = tri.shape[0]
    vl       = np.arange(0, n_vertex // 2)
    vr       = vl + n_vertex // 2
    tl       = np.arange(0, n_tri // 2)
    tr       = tl + n_tri // 2
    return tri,coord,n_vertex,vl,vr,tl,tr

def _get_AAL_map_name(N):
    return f'AAL{N:02d}' + ('remap' if N == 78 else ('comm' if N == 9 else ''))

def _make_AAL_pyvista_mesh(coord, tri, vertex_values, v_idx, t_idx, reindex_offset=0):
    coords                         = coord[v_idx]
    triangles                      = tri[t_idx] - reindex_offset
    faces                          = np.hstack([ np.full((triangles.shape[0], 1), 3), triangles ]).astype(np.int64).ravel()
    mesh                           = pv.PolyData(coords, faces)
    mesh.point_data['Node values'] = vertex_values[v_idx]
    return mesh

def _exists(X):
    return not(type(X) is type(None))

def _fix_discrete_colors(vertex_values,cmap,clim,brain_bg_color):
    # for discrete colors,
    # nodes are assigned to positive integer values (1, 2, ...)
    vertex_values[np.isnan(vertex_values)] = 0
    vertex_values                          = vertex_values.astype(int)
    assert np.all(vertex_values>=0), 'discrete color scheme must only have positive integers in node_values'
    c_mm = (int(np.min(vertex_values)), int(np.max(vertex_values)))
    if _exists(clim):
        if np.isscalar(clim):
            c_mm = (c_mm[0], np.floor(np.abs(clim)))
        else:
            c_mm = ( clim[0] if _exists(clim[0]) else c_mm[0], clim[1] if _exists(clim[1]) else c_mm[1] )
    clim     = np.asarray(c_mm).flatten() + (-0.51,0.50)
    n_labels = clim[1]
    cmap     = _make_discrete_cmap(cmap, int(np.floor(n_labels)), brain_bg_color)
    return vertex_values, cmap, clim

def _fix_continuous_clim(node_values,clim):
    c_mm = (np.min(node_values), np.max(node_values))
    if c_mm[0] == c_mm[1]:
        c_mm = (0,1)
    if _exists(clim):
        if np.isscalar(clim):
            if np.all(node_values <= 0):
                c_mm = (-np.abs(clim),0)
            elif np.all(node_values >= 0):
                c_mm = (0,np.abs(clim))
            else:
                warnings.warn('scalar clim, but node_values has negative and positive value. Assumed to be postive color cutoff. Otherwise, give clim=(min,max)')
                c_mm = (c_mm[0],clim)
        else:
            c_mm = ( clim[0] if _exists(clim[0]) else c_mm[0], clim[1] if _exists(clim[1]) else c_mm[1] )
    return np.asarray(c_mm).flatten()

def _load_AAL_struct(aal=None):
    if not _exists(aal):
        assert os.path.isfile('aal_cortex_map_olf294_fix_v7.mat'), 'file aal_cortex_map_olf294_fix_v7.mat needs to be in the same directory as the running script'
        aal = 'aal_cortex_map_olf294_fix_v7.mat'
    if isinstance(aal,str):
        aal = scipy.io.loadmat(aal,squeeze_me=True)
    return aal

def plot_trisurf_pyvista(
    node_values,
    aal=None,
    cmap='viridis',
    clim=None,
    discrete=False,
    show_edges=False,
    brain_background_color=(1.0, 1.0, 1.0),
    fig_background_color=(1.0, 1.0, 1.0),
    notebook=None,
    panel_w=400,
    panel_h=400
):
    """
    Render cortical surface data on an AAL-based triangulated brain surface
    using PyVista, producing four standard anatomical views.

    The function displays:
        [ Left hemisphere (outer) | Left hemisphere (inner) |
          Right hemisphere (inner) | Right hemisphere (outer) ]

    This function is **rendering-only**:
    - No colorbar is created
    - No Matplotlib logic is involved
    - Intended to be paired with an external (e.g. Matplotlib) colorbar

    Parameters
    ----------
    node_values : array-like, shape (N,)
        Per-node values to be projected onto the cortical surface.
        N must match one of the supported AAL parcellations
        (e.g., 9, 78, 90, 116, etc., depending on available maps).

    aal : dict or str or None, optional
        A loaded MATLAB AAL structure (as returned by scipy.io.loadmat),
        or a path to a .mat file containing it.
        If None, the default file 'aal_cortex_map_olf294_fix_v7.mat'
        is loaded from the current working directory.

    cmap : str or matplotlib colormap, optional
        Colormap used for surface coloring.
        For discrete data, this will be internally converted to a
        categorical colormap with the appropriate number of entries.

    clim : tuple (vmin, vmax) or None, optional
        Color limits for mapping values to colors.
        - For continuous data, defaults to (min(node_values), max(node_values))
        - For discrete data, this is overridden to ensure integer-centered bins

    discrete : bool, optional
        If True, treat node_values as categorical labels.
        This triggers:
        - integer-based color bins
        - categorical colormap handling
        - compatibility with categorical colorbars

    show_edges : bool, optional
        If True, draw triangle edges on the surface meshes.

    brain_background_color : tuple of float, optional
        RGB color used for vertices with no assigned node
        (i.e., background or unlabeled regions).

    fig_background_color : tuple of float, optional
        RGB background color of the PyVista rendering window.

    notebook : bool or None, optional
        Whether to use PyVista's notebook backend.
        Leave as None to let PyVista decide automatically.

    panel_w : int, optional
        Width (in pixels) of each cortical panel.

    panel_h : int, optional
        Height (in pixels) of each cortical panel.

    Returns
    -------
    plotter : pyvista.Plotter
        Configured PyVista plotter containing the four rendered views.
        The caller is responsible for calling `plotter.show()`.

    cmap : matplotlib colormap
        The colormap actually used for rendering.
        For discrete data, this may differ from the input `cmap`
        (e.g., truncated to the number of categories).

    clim : tuple (vmin, vmax)
        The color limits actually used for rendering.
        This is guaranteed to be consistent with the returned colormap.

    Notes
    -----
    - Vertices with node index 0 are treated as background and rendered
      using `brain_background_color`.
    - Camera views and orientations are chosen to match standard
      neuroimaging conventions.
    - This function is designed to be paired with a separate colorbar
      generator (e.g., a Matplotlib-based fake colorbar) for full control
      over ticks, labels, and layout.
    """
    aal = _load_AAL_struct(aal)
    
    node_values    = np.atleast_1d(node_values).flatten()
    N              = len(node_values)

    # separating LR surface vertices / triangles
    tri,coord,n_vertex,vl,vr,tl,tr = _get_surf_AAL_LR_vertices(aal['surf'])
    
    # node -> vertex map
    node_to_vertex = aal['map'][_get_AAL_map_name(N)].item()
    mask           = node_to_vertex > 0

    # Assigning color values to each surface vertex
    vertex_values       = np.full(n_vertex, np.nan)
    vertex_values[mask] = node_values[node_to_vertex[mask]-1] # -1 for 0-based index

    # -------------------------
    # Discrete color handling
    # -------------------------
    if discrete:
        vertex_values, cmap, clim = _fix_discrete_colors(vertex_values,cmap,clim,brain_background_color)
    else:
        clim = _fix_continuous_clim(node_values,clim)

    # Left and right meshes
    # with node_values already encoded
    mesh_L = _make_AAL_pyvista_mesh(coord, tri, vertex_values, vl, tl, reindex_offset=0)
    mesh_R = _make_AAL_pyvista_mesh(coord, tri, vertex_values, vr, tr, reindex_offset=n_vertex // 2)

    # -------------------------
    # Plotter with 4 subplots
    # -------------------------
    fig_w = 4 * panel_w
    fig_h = panel_h

    plotter = pv.Plotter(
        shape=(1, 4),
        window_size=(fig_w, fig_h),
        notebook=notebook,
        border=False
    )
    plotter.set_background(fig_background_color)

    # panel configuration
    #           L outer, L inner,  R inner, R outer
    rolls  = [     90  ,  -90   ,    90   ,   -90   ]
    views  = [ (-90, 0), (90, 0), (-90, 0), (90, 0) ]
    meshes = [  mesh_L , mesh_L ,  mesh_R , mesh_R  ]
    for i, (mesh, (az, el), roll) in enumerate(zip(meshes, views, rolls)):
        plotter.subplot(0, i)
        add_mesh_kwargs = dict(
            scalars='Node values',
            cmap=cmap,
            clim=clim,
            show_edges=show_edges,
            smooth_shading=True,
            nan_color=brain_background_color,
            show_scalar_bar=False
        )
        if discrete:
            #add_mesh_kwargs["cmap"] = cmap
            add_mesh_kwargs["categories"] = True
        plotter.add_mesh(mesh, **add_mesh_kwargs)
        plotter.view_xy()
        plotter.camera.azimuth   = az
        plotter.camera.elevation = el
        plotter.camera.roll      = roll
        plotter.camera.zoom(1.25)
        plotter.remove_bounds_axes()
        plotter.enable_lightkit()

    return plotter, cmap, clim


def create_fake_colorbar(
    plotter,
    cmap,
    clim,
    *,
    fig=None,
    discrete=False,
    tick_positions=None,
    tick_labels=None,
    cbar_text=None,
    width=0.4,
    height=0.05,
    y=0.05,
    fmt='%.2g',
    dpi=150,
    fontsize=12,
    labelsize=10
):
    """
    Create a Matplotlib-based colorbar aligned with a PyVista surface rendering.

    This function generates a "fake" colorbar using Matplotlib that is
    visually and numerically consistent with scalar data rendered in
    PyVista. It is intended as a replacement for PyVista/VTK scalar bars,
    which offer limited control over tick placement, labels, and formatting.

    The colorbar is created in normalized figure coordinates and can be
    positioned precisely (e.g., centered below a multi-panel brain figure).

    Parameters
    ----------
    plotter : pyvista.Plotter
        PyVista plotter used to render the surface. Only the window size
        is used to infer the Matplotlib figure dimensions.

    cmap : str or matplotlib colormap
        Colormap used for the PyVista rendering. The same colormap is
        reused here to ensure visual consistency.

    clim : tuple (vmin, vmax)
        Color limits used for mapping data values to colors in PyVista.

    fig : matplotlib.figure.Figure or None, optional
        Existing Matplotlib figure to draw the colorbar into.
        If None, a new figure is created with dimensions matching the
        PyVista window.

    discrete : bool, optional
        If True, create a categorical (discrete) colorbar using
        integer-centered bins and BoundaryNorm.
        If False, create a continuous colorbar using linear normalization.

    tick_positions : array-like of float, optional
        Numeric positions of colorbar ticks (in data units).
        Required when `discrete=True`.
        For continuous colorbars, this controls explicit tick placement.

    tick_labels : list of str or None, optional
        Text labels to display at each tick position.
        If None, tick labels are generated automatically using `fmt`.

    cbar_label : str or None, optional
        Label for the colorbar axis.

    width : float, optional
        Width of the colorbar as a fraction of the figure width (0–1).

    height : float, optional
        Height of the colorbar as a fraction of the figure height (0–1).

    y : float, optional
        Vertical position of the colorbar (bottom edge), expressed as
        a fraction of the figure height (0–1).

    fmt : str, optional
        printf-style format string used for numeric tick labels
        when `tick_labels` is not provided.

    dpi : int, optional
        Dots per inch used when creating a new Matplotlib figure.

    fontsize : int, optional
        Font size for the colorbar label.

    labelsize : int, optional
        Font size for the colorbar tick labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure containing the colorbar.

    ax : matplotlib.axes.Axes
        The Matplotlib axes object used for the colorbar.

    cbar : matplotlib.colorbar.Colorbar
        The created Matplotlib colorbar object.

    Notes
    -----
    - For discrete colorbars, categories are defined by `tick_positions`,
      and each category occupies an equal-width bin centered on its value.
      This supports background or "null" categories (e.g., value 0).
    - Exact alignment between the PyVista surface colors and the colorbar
      requires that the same colormap and color limits be used in both.
    - This function is designed for publication-quality figures where
      precise control over colorbar layout and labeling is required.
    """

    # -------------------------
    # Figure size from PyVista
    # -------------------------
    win_w, win_h = plotter.window_size
    fig_w = win_w / dpi
    fig_h = win_h / dpi

    if not _exists(fig):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # -------------------------
    # Axes placement (centered)
    # -------------------------
    x0 = 0.5 - width / 2
    ax = fig.add_axes([x0, y, width, height])

    cmap = plt.get_cmap(cmap)

    # -------------------------
    # NORMALIZATION (this is the key)
    # -------------------------
    if discrete:
        if tick_positions is None:
            raise ValueError("Discrete colorbar requires numeric tick_positions")

        tick_positions = np.asarray(tick_positions, dtype=float)

        # Bin edges: centered on each category
        boundaries = np.concatenate([
            [tick_positions[0] - 0.5],
            tick_positions + 0.5
        ])

        norm = mpl.colors.BoundaryNorm(
            boundaries=boundaries,
            ncolors=len(boundaries) - 1,
            clip=True
        )

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            cax=ax,
            orientation='horizontal',
            boundaries=boundaries,
            ticks=tick_positions,
            spacing='uniform'
        )

        # Labels
        if tick_labels is not None:
            if len(tick_labels) != len(tick_positions):
                raise ValueError("tick_labels and tick_positions must match")
            cbar.set_ticklabels(tick_labels)
        else:
            cbar.set_ticklabels([fmt % t for t in tick_positions])

    else:
        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            cax=ax,
            orientation='horizontal',
            ticks=tick_positions
        )

        if tick_positions is not None:
            if tick_labels is not None:
                cbar.set_ticklabels(tick_labels)
            else:
                cbar.set_ticklabels([fmt % t for t in tick_positions])
        else:
            cbar.formatter = mpl.ticker.FormatStrFormatter(fmt)
            cbar.update_ticks()


    # -------------------------
    # Styling
    # -------------------------
    cbar.ax.tick_params(labelsize=labelsize)

    if cbar_text is not None:
        cbar.set_label(cbar_text, fontsize=fontsize)

    return fig, ax, cbar


def _pv_world_to_pixel(plotter, xyz):
    """
    Convert a 3D PyVista world coordinate to pixel coordinates
    using the active renderer.
    """
    x, y, z = map(float, xyz)
    ren = plotter.renderer

    ren.SetWorldPoint(x, y, z, 1.0)
    ren.WorldToDisplay()
    px, py, pz = ren.GetDisplayPoint()
    return px, py, pz


def _plot_subcortical_nodes_symbols_internal(plotter, img, ax, handles, r, v, pSymbol, zlevel, marker_size, cmap, clim, hemi_labels):
    norm = plt.Normalize(*clim)
    for i, (xyz, val) in enumerate(zip(r, v)):
        color = cmap(norm(val))
        marker = pSymbol[i % len(pSymbol)]
        px,py,_ = _pv_world_to_pixel(plotter,xyz)
        py = img.shape[0] - py  # flip Y for imshow
        ms = marker_size
        if marker in 'os':
            ms -= 0.5
        h = ax.plot(
            px,
            py,
            marker=marker,
            linestyle='None',
            markersize=ms,
            markerfacecolor=color,
            markeredgecolor='k',
            markeredgewidth=0.5,
            zorder=100
        )

        handles.append(h)
    return handles

def _plot_subcortical_nodes_symbols(
    plotter,
    img,
    ax,
    aal,
    values,
    N,
    *,
    cmap='viridis',
    clim=None,
    show_symbol_legend=False,
    legend_ax=None,
    marker_size=6,
    zlevel=10
):
    """
    Plot AAL region symbols in 3D using Matplotlib, faithfully reproducing
    the original MATLAB logic.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D Matplotlib axes where the symbols will be plotted.

    aal : dict
        AAL structure loaded from a MATLAB .mat file via scipy.io.loadmat
        (with squeeze_me=True).

    values : array-like, shape (N,)
        Per-node values associated with the AAL parcellation.

    N : int
        Number of AAL nodes (e.g., 9, 90, 306).

    cmap : str or matplotlib colormap, optional
        Colormap used to map values to colors.

    clim : tuple (vmin, vmax) or None, optional
        Color limits. If None, inferred from values.

    show_symbol_legend : bool, optional
        Whether to create a legend mapping symbols to region names.

    legend_ax : matplotlib.axes.Axes or None, optional
        Axes where the legend should be drawn. Required if
        show_symbol_legend=True.

    marker_size : int, optional
        Marker size passed to plt.scatter.

    zlevel : float, optional
        Z-order offset to ensure symbols appear above the surface.

    Returns
    -------
    handles : list
        List of Matplotlib artist handles corresponding to the plotted symbols.
    """
    # -------------------------------------------------
    # Marker symbols
    # -------------------------------------------------
    values = np.asarray(values).flatten()
    cmap = plt.get_cmap(cmap)

    if clim is None:
        clim = (np.nanmin(values), np.nanmax(values))

    # -------------------------------------------------
    # Label selection and setdiff (MATLAB-equivalent)
    # -------------------------------------------------
    if N == 90:
        al = 'AAL90'
        labels_full = np.asarray(aal['labels']['AAL90'].item())
        labels_sub  = np.asarray(aal['labels']['AAL78remap'].item())
    else:
        al = 'AAL306'
        labels_full = np.asarray(aal['labels']['AAL306'].item())
        labels_sub  = np.asarray(aal['labels']['AAL294'].item())

    mask = ~np.isin(labels_full, labels_sub)
    sn   = labels_full[mask]
    indS = np.nonzero(mask)[0]
    # -------------------------------------------------
    # Left / Right hemisphere detection
    # -------------------------------------------------
    kL = np.array(['_L' in s for s in sn])
    kR = np.array(['_R' in s for s in sn])

    pos = np.asarray(aal['pos'][al].item())
    rL = pos[indS[kL], :].copy()
    rR = pos[indS[kR], :].copy()

    # -------------------------------------------------
    # Coordinate displacements (faithful to MATLAB)
    # -------------------------------------------------
    #print({k:s+','+n for k,(s,n) in enumerate(zip(pSymbol,sn[kL]))})
    pSymbol         = list('so^dv<') # match matlab plot
    map_node_to_ind = {'Hip':0, 'Amy':1, 'Cau':2, 'Put':3, 'Pal':4, 'Tha':5}
    rL[:, 0] = 0          # x = 0
    rR[:, 0] = 0          # x = 0
    rL[:, 2] += 20        # z += 20
    rR[:, 2] += 20        # z += 20
    rL[:, 1] -= 10        # y -= 10
    rR[:, 1] -= 10        # y -= 10
    rL[map_node_to_ind['Hip'], 1] += 5
    rR[map_node_to_ind['Hip'], 1] += 5
    rL[map_node_to_ind['Pal'], 1] -= 5
    rR[map_node_to_ind['Pal'], 1] -= 5
    rL[map_node_to_ind['Put'], 2] -= 5
    rR[map_node_to_ind['Put'], 2] -= 5
    rL[map_node_to_ind['Amy'], 1] += 0
    rR[map_node_to_ind['Amy'], 1] += 0
    rL[map_node_to_ind['Cau'], 1] += 2
    rR[map_node_to_ind['Cau'], 1] += 2
    rL[map_node_to_ind['Cau'], 2] -= 2
    rR[map_node_to_ind['Cau'], 2] -= 2

    # -------------------------------------------------
    # Value assignment
    # -------------------------------------------------
    if N == 9:
        label_name = _get_AAL_map_name(N)
        labels     = np.asarray(aal['labels'][label_name].item())

        indS_sub = np.where(
            np.char.lower(labels) == 'subcortical'
        )[0][0]

        ss = rL.shape[0]
        vL = np.full(ss, values[indS_sub])
        vR = np.full(ss, values[indS_sub])
    else:
        vL = values[indS[kL]]
        vR = values[indS[kR]]

    if not np.any(vL):
        vL = np.zeros_like(vL)
    if not np.any(vR):
        vR = np.zeros_like(vR)

    # -------------------------------------------------
    # Plotting (Matplotlib only)
    # -------------------------------------------------
    plotter.subplot(0,1)
    handles = _plot_subcortical_nodes_symbols_internal(plotter,img,ax, []     , rL, vL, pSymbol, zlevel, marker_size, cmap, clim, sn[kL])
    plotter.subplot(0,2)
    handles = _plot_subcortical_nodes_symbols_internal(plotter,img,ax, handles, rR, vR, pSymbol, zlevel, marker_size, cmap, clim, sn[kR])
    # -------------------------------------------------
    # Optional symbol legend
    # -------------------------------------------------
    if show_symbol_legend:
        if legend_ax is None:
            raise ValueError("legend_ax must be provided when show_symbol_legend=True")

        legend_labels = [
            s.replace('_L', '').replace('_', ' ')
            for s in sn[kL]
        ]

        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker=pSymbol[i % len(pSymbol)],
                linestyle='',
                markerfacecolor='w',
                markeredgecolor='k',
                markersize=marker_size
            )
            for i in range(len(legend_labels))
        ]

        legend_ax.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc='lower left', ncol=2,
            bbox_to_anchor=(0,-0.15)
        )
        legend_ax.axis('off')

    return handles


def plot_trisurf_wrapper(
    *args,
    show_subcortical_nodes=True,
    show_subcortical_symbol_leg=True,
    panel_w_in=2.5,     # inches
    panel_h_in=None,     # inches
    dpi=300,
    fig=None,
    ax=None,
    add_colorbar=True,
    colorbar_kwargs=None,
    **kwargs
):
    """
    High-level wrapper that renders a PyVista brain surface at a specified
    physical size and DPI, embeds it into Matplotlib, and optionally adds
    a Matplotlib-based colorbar.

    DPI is used to determine the PyVista rendering resolution, ensuring
    pixel-perfect alignment between PyVista and Matplotlib.
    """
    kwargs['discrete'] = kwargs['discrete'] if 'discrete' in kwargs.keys() else False
    panel_h_in         = panel_h_in         if _exists(panel_h_in)         else panel_w_in
    node_values        = args[0]            if len(args)                   else kwargs['node_values']
    aal                = kwargs['aal']      if 'aal' in kwargs.keys()      else None
    aal                = _load_AAL_struct(aal)
    N                  = len(node_values)
    # -------------------------------------------------
    # 1. Convert physical size → pixels (THE key step)
    # -------------------------------------------------
    panel_w_px = int(round(panel_w_in * dpi))
    panel_h_px = int(round(panel_h_in * dpi))

    # Pass pixel sizes to PyVista renderer
    kwargs["panel_w"] = panel_w_px
    kwargs["panel_h"] = panel_h_px

    # -------------------------------------------------
    # 2. Render with PyVista (pixel-exact)
    # -------------------------------------------------
    plotter, cmap, clim = plot_trisurf_pyvista(*args, **kwargs)

    img = plotter.screenshot(transparent_background=True)
    #img = plotter.show(
    #    interactive=False,
    #    screenshot=True,
    #    auto_close=True
    #)

    h_px, w_px = img.shape[:2]

    # -------------------------------------------------
    # 3. Create Matplotlib figure/axes (same pixels)
    # -------------------------------------------------
    if fig is None:
        fig = plt.figure(
            figsize=(w_px / dpi, h_px / dpi),
            dpi=dpi
        )

    if ax is None:
        ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(img)
    ax.axis("off")

    if show_subcortical_nodes and (N in [9,90,306]):
        _plot_subcortical_nodes_symbols(
            plotter,
            img,
            ax,
            aal,
            node_values,
            N,
            cmap=cmap,
            clim=clim,
            show_symbol_legend=show_subcortical_symbol_leg,
            legend_ax=ax,
            marker_size=6,
            zlevel=10
        )
    
    
    # -------------------------------------------------
    # 4. Optional Matplotlib colorbar
    # -------------------------------------------------
    cbar = None
    if add_colorbar:
        if not _exists(colorbar_kwargs):
            colorbar_kwargs = {}

        _, _, cbar = create_fake_colorbar(
            plotter,
            cmap=cmap,
            clim=clim,
            discrete=kwargs['discrete'],
            fig=fig,
            dpi=dpi,
            **colorbar_kwargs
        )
    
    plotter.close()

    return fig, ax, cbar
