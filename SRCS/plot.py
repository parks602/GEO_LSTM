# Loading all the required libraries.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy import feature
from cmaps import COLORBARS

def plot_learningCurve(args, train_loss, valid_loss):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss,'-m', label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,'-b', label='Validation Loss')
    
    # validation loss의 최저값 지점을 찾기
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0, 0.5) # 일정한 scale
    plt.xlim(0, len(train_loss)+1) # 일정한 scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig('%s/loss_plot.png' %(args.outf), bbox_inches = 'tight')
    plt.close()

def explore_test_loss_over_time(args, y_test, y_pred, criterion):
    loss = np.zeros(y_test.shape[1])
    for i in range(y_test.shape[1]):
        y_test_notna = (y_test.iloc[:,i]).reset_index().drop(['index'],axis=1).dropna()
        idx = y_test_notna.index
        loss[i] = criterion(y_test_notna, np.array(y_pred)[idx,i])
    plt.figure(figsize=(20,6))
    plt.plot(np.arange(len(loss)), loss, c='black', alpha=0.5, label = 'MAE')
    plt.plot(np.arange(len(loss)), np.tile(np.mean(loss), len(loss)), c='red', alpha=0.5, linestyle='--', label='mean error')
    plt.legend()
    plt.title('Test Loss')
    plt.savefig("%s/explore_test_loss.pdf" %(args.outf), dpi=300)
    plt.close()
#
#def plot_coefficient_maps(args, y_pred_mesh):
#    dset, mesh, field, _, _ = read_data(args)
#    u_, s_, v_, _, _, _ = decomposition(args, dset['X_train'], dset['y_train'])
#    timeindex = field.index
#    label = [0, len(timeindex)//2, len(timeindex)-1]
#    #mesh = mesh[ mesh['hgt'] > 0 ]
#
#    plt.figure(figsize=(21,12))
#    plt.subplot(3,3,1)
#    plt.plot(np.arange(len(v_[:,0])), v_[:,0], color='black', alpha=0.5)
#    plt.xticks(label, [timeindex[i] for i in label])
#    plt.title('Temporal basis, first component')
#    plt.subplot(3,3,2)
#    plt.plot(np.arange(len(v_[:,1])), v_[:,1], color='black', alpha=0.5)
#    plt.xticks(label, [timeindex[i] for i in label])
#    plt.title('Temporal basis, second component')
#    plt.subplot(3,3,3)
#    plt.plot(np.arange(len(v_[:,2])), v_[:,2], color='black', alpha=0.5)
#    plt.xticks(label, [timeindex[i] for i in label])
#    plt.title('Temporal basis, third component')
#
#    plt.subplot(3,3,4)
#    plt.scatter(dset['X_train']['lon'], dset['X_train']['lat'],c=u_[:,0], vmin=-1, vmax=4)
#    plt.xticks(rotation=45)
#    plt.colorbar()
#    plt.title('Standardized spatial coefficients, first component')
#    plt.subplot(3,3,5)
#    plt.scatter(dset['X_train']['lon'], dset['X_train']['lat'],c=u_[:,1], vmin=-2, vmax=3)
#    plt.xticks(rotation=45)
#    plt.colorbar()
#    plt.title('Standardized spatial coefficients, second component')
#    plt.subplot(3,3,6)
#    plt.scatter(dset['X_train']['lon'], dset['X_train']['lat'],c=u_[:,2], vmin=-3, vmax=1)
#    plt.xticks(rotation=45)
#    plt.colorbar()
#    plt.title('Standardized spatial coefficients, third component')
#
#    plt.subplot(3,3,7)
#    plt.scatter(mesh['lon'], mesh['lat'],c= y_pred_mesh[:,0], vmin=-1, vmax=4, marker =',', s =1.3)
#    plt.xticks(rotation=45)
#    plt.colorbar()
#    plt.title('Modelled spatial coefficients, first component')
#    plt.subplot(3,3,8)
#    plt.scatter(mesh['lon'], mesh['lat'],c= y_pred_mesh[:,1], vmin=-2, vmax=3, marker =',', s =1.3)
#    
#    plt.xticks(rotation=45)
#    plt.colorbar()
#    plt.title('Modelled spatial coefficients, second component')
#    plt.subplot(3,3,9)
#    plt.scatter(mesh['lon'], mesh['lat'],c= y_pred_mesh[:,2], vmin=-3, vmax=1, marker =',', s =1.3)
#    plt.xticks(rotation=45)
#    plt.colorbar()
#    plt.title('Modelled spatial coefficients, third component')
#    plt.subplots_adjust(hspace = 0.4, wspace = 0.2)
#
#    plt.savefig('%s/Temperature_eofs_allmesh.pdf' %(args.outf), dpi=300)
#    plt.close()
#
#def plot_compare_true_pred(args, y_hat_mesh, y_hat_test):
#    dset, mesh, field, coords, _ = read_data(args)
#    seed = np.random.seed(seed=args.manualSeed)
#    loc = np.random.randint(0, coords['test_coords'].shape[0])
#    loc_x = coords['test_coords']['lon'].iloc[loc]
#    loc_y = coords['test_coords']['lat'].iloc[loc]
#    timestamp = np.random.randint(0, field.shape[1])
#    field = np.transpose(field.values)
#
#    if args.var == "T3H":
#        vmin = -40.
#        vmax = 40.
#    else:
#        vmin = 0
#        vmax = 100.
#    
#    plt.figure(figsize=(13,6))
#    plt.subplots_adjust(wspace = 0.05)
#    plt.subplot(2,2,1)
#    ### True map
#    plt.scatter(coords['field_coords']['lon'], coords['field_coords']['lat'],c= field[:,timestamp], s = 2,  vmin=vmin, vmax=vmax)
#    plt.xticks([], []); plt.yticks([], [])
#    plt.colorbar(label = 'Field Value')
#    plt.scatter(loc_x, loc_y, c ='black', marker='+', s=150)
#    plt.title('True map')
#    plt.subplot(2,2,2)
#    ### Predicted map
#    m = plt.scatter(mesh['lon'], mesh['lat'], c= y_hat_mesh[:,timestamp], s = 2,  vmin=vmin, vmax=vmax)
#    #mesh['v'] = -999.
#    #mesh.loc[mesh['hgt'] > 0, 'v'] = y_hat_mesh[:,timestamp]
#    #re_mesh = mesh[ mesh['v'] > -40 ]
#    #m = plt.scatter(re_mesh['lon'], re_mesh['lat'], s=1, c=re_mesh['v'], vmin=vmin, vmax=vmax)
#    m.cmap.set_under('w')
#    plt.xticks([], []); plt.yticks([], [])
#    plt.colorbar(label = 'Field Value')
#    plt.scatter(loc_x, loc_y, c ='black', marker='+', s=150)
#    plt.title('Predicted map')
#    
#    plt.subplot(212)
#    plt.axvline(timestamp, color='black', linestyle='--', linewidth=1)
#    plt.plot(np.arange(1008), dset['y_test'][loc,:1008], label='True', color = 'black', linewidth = 0.8)
#    plt.plot(np.arange(1008), y_hat_test[loc,:1008], label='Predicted', color = 'orange', linewidth = 1.2)
#    plt.title('True versus predicted time series')
#    plt.ylabel('Field value')
#    plt.xlabel('$t$')
#    plt.legend()
#    plt.savefig("%s/Simdata_predictions_noise_.pdf" %(args.outf), dpi=300)
#    plt.close()

def make_colormaps(var):
    maxc = 255.
    cmaps = COLORBARS()
    c_value, c_over, c_under= cmaps.getColorBar(var)
    colors = np.array(c_value) / maxc
    colors = colors.tolist()
    cms = LinearSegmentedColormap.from_list(var,colors,N=len(colors))
    if c_over is not None:   cms.set_over(np.array(c_over) / maxc)
    if c_under is not None : cms.set_under(np.array(c_under) / maxc)
    return cms

def set_levels(var, dtime):
    if var == "T3H":
        mm = str(dtime)[4:6]
        if int(mm) >= 12 and int(mm) < 3:
            clev = np.linspace(-45,15,61,endpoint=True)
            ticks= np.linspace(-45,15,31,endpoint=True)
        elif int(mm) >= 6 and int(mm) < 9:
            clev = np.linspace(-15,45,61,endpoint=True)
            ticks= np.linspace(-15,45,31,endpoint=True)
        else:
            clev = np.linspace(-30,30,61,endpoint=True)
            ticks= np.linspace(-30,30,31,endpoint=True)
    elif var == "REH":
        clev = np.linspace(0,100,21,endpoint=True)
        ticks= clev
    return clev, ticks

def plot_pred_map(args, y_hat_mesh):
    nrow = 149
    ncol = 253
    _, mesh, field, _, dtime = read_data(args)
    seed = np.random.seed(seed=args.manualSeed)
    timestamp = np.random.randint(0, field.shape[1])
    #field = np.transpose(field.values)
    if len(mesh['lon']) > nrow * ncol:
        nrow = 745
        ncol = 1265

    for i in range(25):
        fig = plt.figure(figsize=(12,10))
        proj= ccrs.LambertConformal(central_longitude=126.,
                                    central_latitude=36.,
                                    standard_parallels=(30,60))
        ax  = plt.axes(projection=proj)
        #ax.set_extent([123.8, 132.2, 31.5, 43])

        ax.add_feature(feature.COASTLINE,linewidth=0.5)

        cms = make_colormaps(args.var)
        clev, ticks = set_levels(args.var, dtime[i])

        lons = np.array(mesh['lon']).reshape(nrow,ncol)
        lats = np.array(mesh['lat']).reshape(nrow,ncol)
        data = y_hat_mesh[:,i].reshape(nrow,ncol)
        maps = ax.contourf(lons, lats, data,
                           transform=ccrs.PlateCarree(),
                           levels=clev, cmap=cms)
        cbar = plt.colorbar(maps, ax=ax, orientation="vertical", ticks=ticks)
        cbar.draw_all()

        if args.var == "T3H":
            plt.title(f"Temperature, {dtime[i]} ")
        else:
            plt.title(f"Relative Humidity, {dtime[i]}")

        plt.tight_layout()
        plt.savefig("%s/pred_%d.png" %(args.outf,i),bbox_inches='tight')
        plt.close()

def plot_grid_diff_map(var, raw, pred, dtime, grid, opath):
    proj= ccrs.LambertConformal(central_longitude=126.,
                                central_latitude=36.,
                                standard_parallels=(30,60))
    lons = grid[:,:,0]
    lats = grid[:,:,1]
    fig, ax = plt.subplots(1,3, figsize=(12,5), subplot_kw=dict(projection=proj))
    # raw
    ax[0].add_feature(feature.COASTLINE,linewidth=0.5)
    cms = make_colormaps(var)
    clev, ticks = set_levels(var, dtime)
    maps = ax[0].contourf(lons, lats, raw,
                        transform=ccrs.PlateCarree(),
                        levels=clev, cmap=cms)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(maps, cax=cax, orientation='vertical', ticks=ticks)
    ax[0].set_title("MODEL")
    # pred 
    ax[1].add_feature(feature.COASTLINE,linewidth=0.5)
    cms = make_colormaps(var)
    clev, ticks = set_levels(var, dtime)
    maps = ax[1].contourf(lons, lats, pred,
                        transform=ccrs.PlateCarree(),
                        levels=clev, cmap=cms)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(maps, cax=cax, orientation='vertical', ticks=ticks)
    ax[1].set_title("PRED")
    # diff
    ax[2].add_feature(feature.COASTLINE,linewidth=0.5)
    cms = 'seismic'
    clev = np.linspace(-10,10,21,endpoint=True)
    ticks= clev
    maps = ax[2].contourf(lons, lats, pred-raw,
                        transform=ccrs.PlateCarree(),
                        levels=clev, cmap=cms)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(maps, cax=cax, orientation='vertical', ticks=ticks)
    ax[2].set_title("DIFF(P-M)")
    fig.suptitle(f"{var}_{dtime}(KST)")
    plt.savefig(f"{opath}/GRD_{var}_{dtime}KST.png", dpi=100, bbox_inches='tight')
    plt.close()

def plot_obs_diff_map(var, mvals, ovals, opnt, dtime, opath, mode):
    proj= ccrs.LambertConformal(central_longitude=126.,
                                central_latitude=36.,
                                standard_parallels=(30,60))
    lons = opnt[:,0]
    lats = opnt[:,1]

    fig, ax = plt.subplots(1,3, figsize=(12,5), subplot_kw=dict(projection=proj))
    # raw
    ax[0].add_feature(feature.COASTLINE,linewidth=0.5)
    cms = make_colormaps(var)
    clev, ticks = set_levels(var, dtime)
    maps = ax[0].scatter(lons, lats, c=mvals, s=0.7,
                        transform=ccrs.PlateCarree(),
                        vmin=np.min(clev), vmax=np.max(clev), cmap=cms)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(maps, cax=cax, orientation='vertical', ticks=ticks)
    ax[0].set_title(f"{mode}_idw")
    # pred
    ax[1].add_feature(feature.COASTLINE,linewidth=0.5)
    cms = make_colormaps(var)
    clev, ticks = set_levels(var, dtime)
    maps = ax[1].scatter(lons, lats, c=ovals, s=0.7,
                        transform=ccrs.PlateCarree(),
                        vmin=np.min(clev), vmax=np.max(clev), cmap=cms)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(maps, cax=cax, orientation='vertical', ticks=ticks)
    ax[1].set_title("OBS")
    # diff
    ax[2].add_feature(feature.COASTLINE,linewidth=0.5)
    #cms = 'bwr'
    cms = 'seismic'
    clev = np.linspace(-10,10,21,endpoint=True)
    ticks= clev
    maps = ax[2].scatter(lons, lats, c=mvals-ovals, s=0.7,
                        transform=ccrs.PlateCarree(),
                        vmin=np.min(clev), vmax=np.max(clev), cmap=cms)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(maps, cax=cax, orientation='vertical', ticks=ticks)
    ax[2].set_title(f"DIFF({mode}-OBS)")
    fig.suptitle(f"{var}_{dtime}(KST)")
    plt.savefig(f"{opath}/OBS_{mode}_{var}_{dtime}KST.png", dpi=100, bbox_inches='tight')
    plt.close()
