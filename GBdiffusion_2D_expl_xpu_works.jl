# julia --project -O3 --check-bounds=no diffusion_2D_expl_xpu.jl
const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf
using FileIO, Colors, StatsBase
#using Pkg
#Pkg.add("JLD")
#Pkg.add("MAT")
using JLD, MAT

gbarr_gray = load("peridotite3_256_256.jld")["data"]
gbarr = Float32.(gbarr_gray)
gbarr = gbarr./maximum(gbarr) 
#gbarr = ones(256,256)
plot(Gray.(gbarr))

     

@parallel function compute_flux!(qTx, qTy, T, Lam, dx, dy)
    @all(qTx) = -@av_xi(Lam)*@d_xi(T)/dx
    @all(qTy) = -@av_yi(Lam)*@d_yi(T)/dy
    return
end

@parallel function compute_update!(T, qTx, qTy, ρCp, dt, dx, dy)
    @inn(T) = @inn(T) - dt/@inn(ρCp)*(@d_xa(qTx)/dx + @d_ya(qTy)/dy)
    return
end

@views function diffusion_2D(; do_visu=false)
    # Physics
    Lx, Ly   = 10.0, 10.0      # domain extent
    λ0       = 1.0           # background heat conductivity
    fac      = 10000         # enhancement factor in GB
    ttot     = 0.002           # total time
    # Numerics
    n        = 2
    nx, ny   = n*128, n*128  # number of grid points
    ndt      = 10            # sparse timestep computation
    nvis     = 50_000       # sparse visualisation
    # Derived numerics
    dx, dy   = Lx/nx, Ly/ny  # grid cell size
    xc, yc   = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    min_dxy2 = min(dx,dy)^2
    # Array initialisation
    #T        = Data.Array(exp.(.-(xc .- Lx/2).^2 .-(yc' .- Ly/2).^2))
    T        = @zeros(nx, ny)
    #T        = T + 0.1.*gbarr
    T[:, ny].= 1
    qTx      = @zeros(nx-1,ny-2)
    qTy      = @zeros(nx-2,ny-1)
    ρCp      = ones(nx, ny)
    mT       = mean(T', dims=2)
    
    #ρCp[((xc.-Lx/3).^2 .+ (yc'.-Ly/3).^2).<1.0].=0.01
    #ρCp      = Data.Array(ρCp)

    #Lam      = λ0 .+ 0.8.*@rand(nx,ny)
    #Lam      = λ0 .+ 0.5.*exp.(.-(xc .- Lx/2).^2 .-(yc' .- Ly*0.9).^2)
    Lam      = λ0 .+ fac.*λ0.*gbarr
    plot(Gray.(Lam))
    opts2 = (background_color = :transparent,foreground_color = :black, aspect_ratio=1, ticks=nothing, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]))
    p1 = contour(xc,yc,Array(gbarr)', c=:red,levels=[0.25, 1.0]; opts2...) 
    savefig("GBarr.png") 
   

    dt       = min_dxy2/maximum(Lam./ρCp)/4.1
    time=0.0; it=0; t_tic=0.0; niter=0
    tout = time
    # Time loop
    while time < ttot
        it += 1
        if (it == 11) t_tic = Base.time(); niter = 0 end
        if (it % ndt == 0) dt = min_dxy2/maximum(Lam./ρCp)/4.1 end # done every ndt to improve perf
        @parallel compute_flux!(qTx, qTy, T, Lam, dx, dy)
        @parallel compute_update!(T, qTx, qTy, ρCp, dt, dx, dy)

        # add in a Dirichlet BC at top wall and no flux at edges, 
        # just for fun
        T[:, ny] .= 1.0
        T[1, :]  = T[2, :]
        T[nx, :] = T[nx-1, :] 
        #T[:, 1] = T[:, 2]

        niter += 1
        time += dt
        if do_visu && (it % nvis == 0)
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), #=clims=(0.0, 1.0),=# c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            p1 = display(heatmap(xc, yc, Array(T)'; opts...))
            savefig("./figures/$time.png")
            # save an array containing the row (x) means of T
            mT = hcat(mT, mean(T', dims=2))
            tout = hcat(tout,time)
        end
        
    end

    savefig("./figures/$time.png")
    
    file = matopen("mT.mat","w")
    write(file, "mT", mT)
    close(file)

    file = matopen("tout.mat","w")
    write(file, "tout", tout)
    close(file)

  #  file = matopen("yc.mat","w")
  #  write(file, "yc", yc')
  #  close(file)

 #   t_toc = Base.time() - t_tic
 #   @printf("Computed %d steps, physical time = %1.3f\n", it, time)
 #   A_eff = 4/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
 #   t_it  = t_toc/niter                  # Execution time per iteration [s]
 #   T_eff = A_eff/t_it                   # Effective memory throughput [GB/s]
#    @printf("Perf: time = %1.3f sec, T_eff = %1.2f GB/s\n", t_toc, round(T_eff, sigdigits=3))
   return
end

diffusion_2D(; do_visu=true)




