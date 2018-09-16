using UnicodePlots
using SparseArrays

grid = JuAFEM.generate_grid(Quadrilateral, (20,20))

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

heatelement = GenericElement(:heatelement, Quadrilateral)
push!(heatelement, Field(:T, default_interpolation(Quadrilateral),1))

dh = DofHandler(grid)
push_element!(dh, heatelement, Set(collect(1:length(grid.cells)))) # Add a temperature field
close!(dh)
@show dh.cell_dofs
@show dh.cell_dofs_offset
@show grid.cells[1].nodes
@show grid.cells[2].nodes
@show getcoordinates(grid,1)
@show getcoordinates(grid,2)
dbcs = ConstraintHandler(dh)
dbc = Dirichlet(:T, heatelement, union(getfaceset(grid, "left"), getfaceset(grid, "right"), getfaceset(grid, "top"), getfaceset(grid, "bottom")), (x,t)->0)
add!(dbcs, dbc)
close!(dbcs)
update!(dbcs, 0.0)

@time K = JuAFEM.create_sparsity_pattern(dh);
fill!(K.nzval, 1.0);
spy(K)

function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler)
    b = 1.0
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    
    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh,1))

    fe = zeros(n_basefuncs) # Local force vector
    Ke = zeros(n_basefuncs, n_basefuncs) # Local stiffness mastrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh, heatelement))
        fill!(Ke, 0)
        fill!(fe, 0)
        
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δT = shape_value(cellvalues, q_point, i)
                ∇δT = shape_gradient(cellvalues, q_point, i)
                fe[i] += (δT * b) * dΩ
                for j in 1:n_basefuncs
                    ∇T = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇δT ⋅ ∇T) * dΩ
                end
            end
        end
        
        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end;

@time K, f = doassemble(cellvalues, K, dh);
@time apply!(K, f, dbcs)
@time T = K \ f;
vtkfile = vtk_grid("heat", dh)
vtk_point_data(vtkfile, dh, T)
vtk_save(vtkfile);

norm(T)

