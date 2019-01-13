
# # TODO: Maybe nice to add a field like this instead of manually pushing stuff to the dofhandler
struct Field
    name::Symbol
    interpolation::Interpolation
    dim::Int
end

abstract type AbstractElement end

struct Element{T} <: AbstractElement
    name::Symbol
    fields::Vector{Field}
    bcvalues::Vector{BCValues{T}}
    celltype::Type{<:Cell}
end

function Element(name::Symbol, celltype::Type{C}) where {C}
    return Element(Float64, name, celltype)
end

function Element(::Type{T}, name::Symbol, celltype::Type{C}) where {C,T}
    return Element(name, Field[], BCValues{T}[], celltype)
end

function Base.push!(el::AbstractElement, field::Field)
    #checkstuff
    push!(el.bcvalues, BCValues(field.interpolation, default_interpolation(el.celltype)))
    push!(el.fields, field)
end

function Base.show(io::IO, el::AbstractElement)
    println("AbstractElement $(typeof(el))")
end

function get_field(el::AbstractElement, field_name::Symbol)
    j = find_field(el, field_name)
    return j == nothing ? nothing : el.fields[j]
end

function get_bcvalue(el::AbstractElement, field_name::Symbol)
    if isdefined(el, :bcvalues)
        return el.bcvalues[find_field(el, field_name)]
    else
        field = get_field(el,field_name)
        
        return BCValues(field.interpolation, default_interpolation(celltype(el)))
    end
end

function find_field(el::AbstractElement, field_name::Symbol)
    j = findfirst(i->i.name == field_name, el.fields)
    j == 0 && error("did not find field $field_name")
    return j
end

function field_offset(el::AbstractElement, field_name::Symbol)
    offset = 0
    for i in 1:find_field(el, field_name)-1
        field = el.fields[i]
        offset += getnbasefunctions(field.interpolation)::Int * field.dims
    end
    return offset
end

function dof_range(element::AbstractElement, field_name::Symbol)
    f = find_field(element, field_name)
    offset = field_offset(element, field_name)
    n_field_dofs = getnbasefunctions(element.fields[f].interpolation)::Int * element.fields[f].dim
    return (offset+1):(offset+n_field_dofs)
end

ndofs(el::AbstractElement) = sum([getnbasefunctions(field.interpolation)::Int * field.dim for field in el.fields])
nnodes(el::AbstractElement) = nnodes(el.celltype)
nfaces(el::AbstractElement) = nfaces(el.celltype)
celltype(el::AbstractElement) = el.celltype

"""
    DofHandler(grid::Grid)

Construct a `DofHandler` based on the grid `grid`.
"""
struct DofHandler{dim,T}
    
    elements::Vector{AbstractElement}
    elementcells::Vector{Set{Int}}
    cellmapper::Vector{Int} #cellid to elementtype

    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}

    cell_nodes::Vector{Int}
    cell_nodes_offset::Vector{Int}

    cell_coords::Vector{Vec{dim,T}}
    cell_coords_offset::Vector{Int}

    closed::ScalarWrapper{Bool}
    grid::Grid{dim,T}
end

function DofHandler(grid::Grid{dim,T}) where {dim,T}
    DofHandler(AbstractElement[], Set{Int}[], Int[], Int[], Int[], Int[], Int[], Vec{dim,T}[], Int[], ScalarWrapper(false), grid)
end

function Base.show(io::IO, dh::DofHandler)
    println(io, "DofHandler")
    for el in dh.elements
        println(io, "Element $(typeof(el))")
        for field in el.fields
            println(io, "  Field: $(field.name), $(field.interpolation), $(field.dim)")
        end
    end

    if !isclosed(dh)
        print(io, "Not closed!")
    else
        print(io, "Total dofs: ", ndofs(dh))
    end
end

# TODO: This is not very nice, worth storing ndofs explicitly?
#       How often do you actually call ndofs though...
ndofs(dh::DofHandler) = maximum(dh.cell_dofs)
ndofs_per_cell(dh::DofHandler, cell::Int=1) = dh.cell_dofs_offset[cell+1] - dh.cell_dofs_offset[cell]
nnodes_per_cell(dh::DofHandler, cell::Int=1) = dh.cell_coords_offset[cell+1] - dh.cell_coords_offset[cell]
isclosed(dh::DofHandler) = dh.closed[]
ndim(dh::DofHandler, field_name::Symbol) = dh.field_dims[find_field(dh, field_name)]


# Calculate the offset to the first local dof of a field
function get_elementcells(dh::DofHandler, element::AbstractElement)
    j = findfirst(i->i == element, dh.elements)
    j == 0 && error("...")
    return dh.elementcells[j]
end

"""
    dof_range(dh:DofHandler, field_name)

Return the local dof range for `field_name`. Example:

```jldoctest
julia> grid = generate_grid(Triangle, (3, 3));

julia> dh = DofHandler(grid); push!(dh, :u, 3); push!(dh, :p, 1); close!(dh);

julia> dof_range(dh, :u)
1:9

julia> dof_range(dh, :p)
10:12
```
"""
function dof_range(dh::DofHandler, field_name::Symbol)
    f = find_field(dh, field_name)
    offset = field_offset(dh, field_name)
    n_field_dofs = getnbasefunctions(dh.field_interpolations[f])::Int * dh.field_dims[f]
    return (offset+1):(offset+n_field_dofs)
end

#function Base.push!
function push_element!(dh::DofHandler, element::E, cellset::Set{Int}) where E<:AbstractElement
    @assert !isclosed(dh)
    @assert isdefined(element, :fields)
    @assert isdefined(element, :celltype)
    #@assert isdefined(element, :bcvalues)

    push!(dh.elements, element)
    push!(dh.elementcells, cellset)
    return dh
end

# sort and return true (was already sorted) or false (if we had to sort)
function sortedge(edge::Tuple{Int,Int})
    a, b = edge
    a < b ? (return (edge, true)) : (return ((b, a), false))
end

sortface(face::Tuple{Int,Int}) = minmax(face[1], face[2])
function sortface(face::Tuple{Int,Int,Int})
    a, b, c = face
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)
    return (a, b, c)
end

# close the DofHandler and distribute all the dofs
function close!(dh::DofHandler{dim}) where {dim}
    @assert !isclosed(dh)
    
    unique_fields = Vector{Symbol}()
    for element in dh.elements
        append!(unique_fields, [field.name for field in element.fields])
    end
    unique_fields = unique(unique_fields)

    # `vertexdict` keeps track of the visited vertices. We store the global vertex
    # number and the first dof we added to that vertex.
    vertexdicts = [Dict{Int,Int}() for _ in 1:length(unique_fields)]

    # `edgedict` keeps track of the visited edges, this will only be used for a 3D problem
    # An edge is determined from two vertices, but we also need to store the direction
    # of the first edge we encounter and add dofs too. When we encounter the same edge
    # the next time we check if the direction is the same, otherwise we reuse the dofs
    # in the reverse order
    edgedicts = [Dict{Tuple{Int,Int},Tuple{Int,Bool}}() for _ in 1:length(unique_fields)]

    # `facedict` keeps track of the visited faces. We only need to store the first dof we
    # added to the face; if we encounter the same face again we *always* reverse the order
    # In 2D a face (i.e. a line) is uniquely determined by 2 vertices, and in 3D a
    # face (i.e. a surface) is uniquely determined by 3 vertices.
    facedicts = [Dict{NTuple{dim,Int},Int}() for _ in 1:length(unique_fields)]

    # celldofs are never shared between different cells so there is no need
    # for a `celldict` to keep track of which cells we have added dofs too.

    # not implemented yet: more than one facedof per face in 3D
    #dim == 3 && @assert(!any(x->x.nfacedofs > 1, interpolation_infos))

    nextdof = 1 # next free dof to distribute

    #Create mapper cellid -> elementtype id
    resize!(dh.cellmapper, getncells(dh.grid))
    for (elementid, cellset) in enumerate(dh.elementcells)
        for cellid in cellset
            dh.cellmapper[cellid] = elementid#dh.elements[elementid]
        end
    end

    # Calculate cell_dofs_offset beforehand because we no longer loop over the 
    # cells in order (1,2,3,4...)
    push!(dh.cell_dofs_offset, 1) # dofs for the first cell start at 1
    for cellid in 1:getncells(dh.grid)
        element = cellelement(dh, cellid)
        push!(dh.cell_dofs_offset, dh.cell_dofs_offset[cellid]+JuAFEM.ndofs(element))
    end
    
    celldofs_length = sum([JuAFEM.ndofs(e)*length(ec) for (e,ec) in zip(dh.elements, dh.elementcells)])
    resize!(dh.cell_dofs, celldofs_length)
    # loop over all the cells, and distribute dofs for all the fields
    #for (ci, cell) in enumerate(getcells(dh.grid))
    for (element, cellset) in zip(dh.elements, dh.elementcells)
        for ci in sort(collect(cellset))
            cell = dh.grid.cells[ci]

            celldof_counter = dh.cell_dofs_offset[ci]
            @debug println("cell #$ci")
            for field in element.fields

                fi = findfirst(i->i==field.name, unique_fields)
                interpolation_info = InterpolationInfo(field.interpolation)
                dim == 3 && @assert(!(interpolation_info.nfacedofs > 1))

                @debug println("  field: $(field.name)")
                if interpolation_info.nvertexdofs > 0
                    for vertex in vertices(cell)
                        @debug println("    vertex#$vertex")
                        token = Base.ht_keyindex2!(vertexdicts[fi], vertex)
                        
                        if token > 0 # haskey(vertexdicts[fi], vertex) # reuse dofs
                            reuse_dof = vertexdicts[fi].vals[token] # vertexdicts[fi][vertex]
                            for d in 1:field.dim
                                @debug println("      reusing dof #$(reuse_dof + (d-1))")
                                dh.cell_dofs[celldof_counter] = reuse_dof + (d-1)
                                celldof_counter+=1
                                #push!(dh.cell_dofs, reuse_dof + (d-1))
                            end
                        else # token <= 0, distribute new dofs
                            for vertexdof in 1:interpolation_info.nvertexdofs
                                Base._setindex!(vertexdicts[fi], nextdof, vertex, -token) # vertexdicts[fi][vertex] = nextdof
                                for d in 1:field.dim
                                    @debug println("      adding dof#$nextdof")
                                    dh.cell_dofs[celldof_counter] = nextdof
                                    celldof_counter+=1
                                    #push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    end # vertex loop
                end
                if dim == 3 # edges only in 3D
                    if interpolation_info.nedgedofs > 0
                        for edge in edges(cell)
                            sedge, dir = sortedge(edge)
                            @debug println("    edge#$sedge dir: $(dir)")
                            token = Base.ht_keyindex2!(edgedicts[fi], sedge)
                            if token > 0 # haskey(edgedicts[fi], sedge), reuse dofs
                                startdof, olddir = edgedicts[fi].vals[token] # edgedicts[fi][sedge] # first dof for this edge (if dir == true)
                                for edgedof in (dir == olddir ? (1:interpolation_info.nedgedofs) : (interpolation_info.nedgedofs:-1:1))
                                    for d in 1:field.dim
                                        reuse_dof = startdof + (d-1) + (edgedof-1)*field.dim
                                        @debug println("      reusing dof#$(reuse_dof)")
                                        dh.cell_dofs[celldof_counter] = reuse_dof
                                        celldof_counter+=1
                                        #push!(dh.cell_dofs, reuse_dof)
                                    end
                                end
                            else # token <= 0, distribute new dofs
                                Base._setindex!(edgedicts[fi], (nextdof, dir), sedge, -token) # edgedicts[fi][sedge] = (nextdof, dir),  store only the first dof for the edge
                                for edgedof in 1:interpolation_info.nedgedofs
                                    for d in 1:field.dim
                                        @debug println("      adding dof#$nextdof")
                                        dh.cell_dofs[celldof_counter] = nextdof
                                        celldof_counter+=1
                                        #push!(dh.cell_dofs, nextdof)
                                        nextdof += 1
                                    end
                                end
                            end
                        end # edge loop
                    end
                end
                if interpolation_info.nfacedofs > 0 # nfacedofs(interpolation) > 0
                    for face in faces(cell)
                        sface = sortface(face) # TODO: faces(cell) may as well just return the sorted list
                        @debug println("    face#$sface")
                        token = Base.ht_keyindex2!(facedicts[fi], sface)
                        if token > 0 # haskey(facedicts[fi], sface), reuse dofs
                            startdof = facedicts[fi].vals[token] # facedicts[fi][sface]
                            for facedof in interpolation_info.nfacedofs:-1:1 # always reverse (YOLO)
                                for d in 1:field.dim
                                    reuse_dof = startdof + (d-1) + (facedof-1)*field.dim
                                    @debug println("      reusing dof#$(reuse_dof)")
                                    dh.cell_dofs[celldof_counter] = reuse_dof
                                    celldof_counter+=1
                                    #push!(dh.cell_dofs, reuse_dof)
                                end
                            end
                        else # distribute new dofs
                            Base._setindex!(facedicts[fi], nextdof, sface, -token)# facedicts[fi][sface] = nextdof,  store the first dof for this face
                            for facedof in 1:interpolation_info.nfacedofs
                                for d in 1:field.dim
                                    @debug println("      adding dof#$nextdof")
                                    dh.cell_dofs[celldof_counter] = nextdof
                                    celldof_counter+=1
                                    #push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    end # face loop
                end
                if interpolation_info.ncelldofs > 0 # always distribute new dofs for cell
                    @debug println("    cell#$ci")
                    for celldof in 1:interpolation_info.ncelldofs
                        for d in 1:field.dim
                            @debug println("      adding dof#$nextdof")
                            dh.cell_dofs[celldof_counter] = nextdof
                            celldof_counter+=1
                            #push!(dh.cell_dofs, nextdof)
                            nextdof += 1
                        end
                    end # cell loop
                end
            end # field loop
            # push! the first index of the next cell to the offset vector
            #push!(dh.cell_dofs_offset, length(dh.cell_dofs)+1)
        end
    end # cell loop

    #Create coords vector
    push!(dh.cell_coords_offset, 1)
    push!(dh.cell_nodes_offset, 1)
    for cell in dh.grid.cells
        for nodeid in cell.nodes
            push!(dh.cell_nodes, nodeid)
            push!(dh.cell_coords, dh.grid.nodes[nodeid].x)
        end
        push!(dh.cell_nodes_offset, length(dh.cell_nodes)+1)
        push!(dh.cell_coords_offset, length(dh.cell_coords)+1)
    end

    dh.closed[] = true
    return dh
end

function cellelement(dh::DofHandler, cellid::Int)
    return dh.elements[dh.cellmapper[cellid]]
end

function cellnodes!(nodes::Vector{Int}, dh::DofHandler{dim,T}, i::Int) where {dim,T}
    @assert isclosed(dh)
    @assert length(nodes) == nnodes_per_cell(dh, i)
    unsafe_copyto!(nodes, 1, dh.cell_nodes, dh.cell_nodes_offset[i], length(nodes))
    return nodes
end

function cellcoords!(coords::Vector{Vec{dim,T}}, dh::DofHandler{dim,T}, i::Int) where {dim,T}
    @assert isclosed(dh)
    @assert length(coords) == nnodes_per_cell(dh, i)
    unsafe_copyto!(coords, 1, dh.cell_coords, dh.cell_coords_offset[i], length(coords))
    return coords
end

function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

function celldofs(dh::DofHandler, i::Int)
    global_dofs = zeros(Int, ndofs_per_cell(dh,i))
    celldofs!(global_dofs, dh, i)
    return global_dofs
end

# Creates a sparsity pattern from the dofs in a DofHandler.
# Returns a sparse matrix with the correct storage pattern.
"""
    create_sparsity_pattern(dh::DofHandler)

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

See the [Sparsity Pattern](@ref) section of the manual.
"""
@inline create_sparsity_pattern(dh::DofHandler) = _create_sparsity_pattern(dh, false)

"""
    create_symmetric_sparsity_pattern(dh::DofHandler)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
@inline create_symmetric_sparsity_pattern(dh::DofHandler) = Symmetric(_create_sparsity_pattern(dh, true), :U)

function _create_sparsity_pattern(dh::DofHandler, sym::Bool)
    ncells = getncells(dh.grid)
    N = 10^2 * ncells #some random guess
    #N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    #N += ndofs(dh) # always add the diagonal elements
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)
    global_dofs = Int[]
    cnt = 0
    for element_id in 1:ncells
        n = ndofs_per_cell(dh, element_id)
        resize!(global_dofs, n)

        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            cnt += 1
            if cnt > length(J)
                resize!(I, trunc(Int, length(I) * 1.5))
                resize!(J, trunc(Int, length(J) * 1.5))
            end
            I[cnt] = dofi
            J[cnt] = dofj
        end
    end
    @inbounds for d in 1:ndofs(dh)
        cnt += 1
        if cnt > length(J)
            resize!(I, trunc(Int, length(I) + ndofs(dh)))
            resize!(J, trunc(Int, length(J) + ndofs(dh)))
        end
        I[cnt] = d
        J[cnt] = d
    end
    resize!(I, cnt)
    resize!(J, cnt)
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K
end

# dof renumbering
"""
    renumber!(dh::DofHandler, perm)

Renumber the degrees of freedom in the DofHandler according to the
permuation `perm`.

!!! warning
    Remember to do renumbering *before* adding boundary conditions,
    otherwise the mapping for the dofs will be wrong.
"""
function renumber!(dh::DofHandler, perm::AbstractVector{<:Integer})
    @assert isperm(perm) && length(perm) == ndofs(dh)
    cell_dofs = dh.cell_dofs
    for i in eachindex(cell_dofs)
        cell_dofs[i] = perm[cell_dofs[i]]
    end
    return dh
end

WriteVTK.vtk_grid(filename::AbstractString, dh::DofHandler) = vtk_grid(filename, dh.grid)

# Exports the FE field `u` to `vtkfile`
function WriteVTK.vtk_point_data(vtkfile, dh::DofHandler, u::Vector)
    for (ie, element) in enumerate(dh.elements)
        for field in element.fields
            field_dim = field.dim
            space_dim = field_dim == 2 ? 3 : field_dim
            offset = field_offset(element, field.name)
            data = fill(0.0, space_dim, getnnodes(dh.grid))
            for cell in CellIterator(dh, element)
                _celldofs = celldofs(cell)
                counter = 1            
                for node in getnodes(cell)
                    for d in 1:field.dim
                        data[d, node] = u[_celldofs[counter + offset]]
                        @debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                        counter += 1
                    end
                end
            end
            vtk_point_data(vtkfile, data, string(field.name))
        end
    end
    #=
    for f in 1:nfields(dh)
        @debug println("exporting field $(dh.field_names[f])")
        field_dim = dh.field_dims[f]
        space_dim = field_dim == 2 ? 3 : field_dim
        data = fill(0.0, space_dim, getnnodes(dh.grid))
        offset = field_offset(dh, dh.field_names[f])
        for cell in CellIterator(dh)
            _celldofs = celldofs(cell)
            counter = 1
            for node in getnodes(cell)
                for d in 1:dh.field_dims[f]
                    data[d, node] = u[_celldofs[counter + offset]]
                    @debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                    counter += 1
                end
            end
        end
        vtk_point_data(vtkfile, data, string(dh.field_names[f]))
    end
    =#
    return vtkfile
end
