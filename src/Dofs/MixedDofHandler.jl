
mutable struct FieldHandler
    fields::Vector{Field}
    cellset::Set{Int}
    bc_values::Vector{BCValues} # for boundary conditions
end
"""
    FieldHandler(fields::Vector{Field}, cellset)

Construct a `FieldHandler` based on an array of `Field`s and assigns it a set of cells.
"""
function FieldHandler(fields::Vector{Field}, cellset)
    # TODO for now, only accept isoparamtric mapping
    bc_values = [BCValues(field.interpolation, field.interpolation) for field in fields]
    FieldHandler(fields, cellset, bc_values)
end

struct CellVector{T}
    values::Vector{T}
    offset::Vector{Int}
    length::Vector{Int}
end

function Base.getindex(elvec::CellVector, el::Int)
    offset = elvec.offset[el]
    return elvec.values[offset:offset + elvec.length[el]-1]
 end

struct MixedDofHandler{dim,C,T} <: JuAFEM.AbstractDofHandler
    fieldhandlers::Vector{FieldHandler}
    cell_dofs::CellVector{Int}
    cell_nodes::CellVector{Int}
    cell_coords::CellVector{Vec{dim,T}}
    closed::ScalarWrapper{Bool}
    grid::Grid{dim,C,T}
    ndofs::ScalarWrapper{Int}
end

function MixedDofHandler(grid::Grid{dim,C,T}) where {dim,C,T}
    MixedDofHandler{dim,C,T}(FieldHandler[], CellVector(Int[],Int[],Int[]), CellVector(Int[],Int[],Int[]), CellVector(Vec{dim,T}[],Int[],Int[]), JuAFEM.ScalarWrapper(false), grid, JuAFEM.ScalarWrapper(-1))
end

getfieldnames(fh::FieldHandler) = [field.name for field in fh.fields]
getfielddims(fh::FieldHandler) = [field.dim for field in fh.fields]
getfieldinterpolations(fh::FieldHandler) = [field.interpolation for field in fh.fields]
ndofs_per_cell(dh::MixedDofHandler, cell::Int=1) = dh.cell_dofs.length[cell]
nnodes_per_cell(dh::MixedDofHandler, cell::Int=1) = dh.cell_nodes.length[cell]


function celldofs!(global_dofs::Vector{Int}, dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copyto!(global_dofs, 1, dh.cell_dofs.values, dh.cell_dofs.offset[i], length(global_dofs))
    return global_dofs
end

function celldofs(dh::MixedDofHandler, i::Int)
    @assert isclosed(dh)
    return dh.cell_dofs[i]
end

function cellcoords!(global_coords::Vector{Vec{dim,T}}, dh::MixedDofHandler, i::Int) where {dim,T}
    @assert isclosed(dh)
    @assert length(global_coords) == nnodes_per_cell(dh, i)
    unsafe_copyto!(global_coords, 1, dh.cell_coords.values, dh.cell_coords.offset[i], length(global_coords))
    return global_coords
end

function cellnodes!(global_nodes::Vector{Int}, dh::MixedDofHandler, i::Int) where {dim,T}
    @assert isclosed(dh)
    @assert length(global_nodes) == nnodes_per_cell(dh, i)
    unsafe_copyto!(global_nodes, 1, dh.cell_nodes.values, dh.cell_nodes.offset[i], length(global_nodes))
    return global_nodes
end


"""
    getfieldnames(dh::MixedDofHandler)

Returns the union of all the fields. Can be used as an iterable over all the fields in the problem.
"""
function getfieldnames(dh::MixedDofHandler)
    fieldnames = Vector{Symbol}()
    for fh in dh.fieldhandlers
        for name in getfieldnames(fh)
            push!(fieldnames, name)
        end
    end
    return unique!(fieldnames)
end

"""
    getfielddim(dh::MixedDofHandler, name::Symbol)

Returns the dimension of a specific field, given by name. Note that it will return the dimension of the first field found among the `FieldHandler`s.
"""
function getfielddim(dh::MixedDofHandler, name::Symbol)

    for fh in dh.fieldhandlers
        field_pos = findfirst(i->i == name, getfieldnames(fh))
        if field_pos > 0
            return fh.fields[field_pos].dim
        end
    end
    error("did not find field $name")
end


"""
    nfields(dh::MixedDofHandler)

Returns the number of unique fields defined.
"""
nfields(dh::MixedDofHandler) = length(getfieldnames(dh))

function Base.push!(dh::MixedDofHandler, fh::FieldHandler)
    @assert !isclosed(dh)
    push!(dh.fieldhandlers, fh)
    return dh
end

function Base.push!(dh::MixedDofHandler, name::Symbol, dim::Int)
    celltype = getcelltype(dh.grid)
    isconcretetype(celltype) || error("If you have more than one celltype in Grid, you must use push!(dh::MixedDofHandler, fh::FieldHandler)")
    push!(dh, name, dim, default_interpolation(celltype))
end

function Base.push!(dh::MixedDofHandler, name::Symbol, dim::Int, ip::Interpolation)
    @assert !isclosed(dh)

    if length(dh.fieldhandlers) == 0
        cellset = Set(1:getncells(dh.grid))
        push!(dh.fieldhandlers, FieldHandler(Field[],cellset,BCValues[]))
    elseif length(dh.fieldhandlers) > 1
        error("If you have more than one FieldHandler, you must specify field")
    end
    fh = first(dh.fieldhandlers)

    field = Field(name,ip,dim)
    bc_value = BCValues(field.interpolation, field.interpolation)

    push!(fh.fields, field)
    push!(fh.bc_values, bc_value)

    return dh
end

"""
    close!(dh::MixedDofHandler)

Closes the dofhandler and creates degrees of freedom for each cell.
Dofs are created in the following order: Go through each FieldHandler in the order they were added. For each field in the FieldHandler, create dofs for the cell. This means that dofs on a particular cell will be numbered according to the fields; first dofs for field 1, then field 2, etc.
"""
function close!(dh::MixedDofHandler{dim}) where {dim}

    @assert !JuAFEM.isclosed(dh)
    field_names = JuAFEM.getfieldnames(dh)  # all the fields in the problem
    numfields =  length(field_names)

    # Create dicts that stores created dofs
    # Each key should uniquely identify the given type
    vertexdicts = [Dict{Int, Int}() for _ in 1:numfields]
    edgedicts = [Dict{Tuple{Int,Int}, Int}() for _ in 1:numfields]
    facedicts = [Dict{Tuple{Int,Int}, Int}() for _ in 1:numfields]
    celldicts = [Dict{Int, Int}() for _ in 1:numfields]

    # Set initial values
    nextdof = 1  # next free dof to distribute

    append!(dh.cell_dofs.offset, zeros(getncells(dh.grid)))
    append!(dh.cell_dofs.length, zeros(getncells(dh.grid)))

    @debug "\n\nCreating dofs\n"
    for fh in dh.fieldhandlers
        # sort the cellset since we want to loop through the cells in a fixed order
        cellnumbers = sort(collect(fh.cellset))
        nextdof = _close!(
            dh,
            cellnumbers,
            JuAFEM.getfieldnames(fh),
            JuAFEM.getfielddims(fh),
            JuAFEM.getfieldinterpolations(fh),
            nextdof,
            vertexdicts,
            edgedicts,
            facedicts,
            celldicts)
    end
    dh.ndofs[] = maximum(dh.cell_dofs.values)
    dh.closed[] = true

    #Create cell_nodes and cell_coords (similar to cell_dofs)
    push!(dh.cell_nodes.offset, 1)
    push!(dh.cell_coords.offset, 1)
    for cell in dh.grid.cells
        for nodeid in cell.nodes
            push!(dh.cell_nodes.values, nodeid)
            push!(dh.cell_coords.values, dh.grid.nodes[nodeid].x)
        end
        push!(dh.cell_nodes.offset, length(dh.cell_nodes.values)+1)
        push!(dh.cell_coords.offset, length(dh.cell_coords.values)+1)
        push!(dh.cell_nodes.length, length(cell.nodes))
        push!(dh.cell_coords.length, length(cell.nodes))
    end

    return dh
end

function _close!(dh::MixedDofHandler{dim}, cellnumbers, field_names, field_dims, field_interpolations, nextdof, vertexdicts, edgedicts, facedicts, celldicts) where {dim}

    ip_infos = JuAFEM.InterpolationInfo[]
    for interpolation in field_interpolations
        ip_info = JuAFEM.InterpolationInfo(interpolation)
        push!(ip_infos, ip_info)
    end

    # loop over all the cells, and distribute dofs for all the fields
    for ci in cellnumbers
        dh.cell_dofs.offset[ci] = length(dh.cell_dofs.values)+1

        cell = dh.grid.cells[ci]
        cell_dofs = Int[]  # list of global dofs for each cell
        @debug "Creating dofs for cell #$ci"

        for fi in 1:length(field_names)
            @debug "\tfield: $(field_names[fi])"
            ip_info = ip_infos[fi]

            if ip_info.nvertexdofs > 0
                sortfunc = (vertex) -> vertex
                nextdof = _add_dofs(cell_dofs, cell, vertexdicts[fi], field_dims[fi], ip_info.nvertexdofs, nextdof, JuAFEM.vertices, sortfunc)
            end
                   
            if ip_info.nedgedofs > 0 && dim == 3 #Edges only in 3d
                sortfunc = minmax
                nextdof = _add_dofs(cell_dofs, cell, edgedicts[fi], field_dims[fi], ip_info.nedgedofs, nextdof, JuAFEM.edges, sortfunc)
            end

            if ip_info.nfacedofs > 0 && (ip_info.dim == dim)
                sortfunc = JuAFEM.sortface
                nextdof = _add_dofs(cell_dofs, cell, facedicts[fi], field_dims[fi], ip_info.nfacedofs, nextdof, JuAFEM.faces, sortfunc)
            end
            
            if ip_info.ncelldofs > 0
                sortfunc = (cellid) -> cellid
                nextdof = _add_dofs(cell_dofs, ci, celldicts[fi], field_dims[fi], ip_info.ncelldofs, nextdof, (cell) -> ci, sortfunc)
            end

        end
        # after done creating dofs for the cell, push them to the global list
        push!(dh.cell_dofs.values, cell_dofs...)
        dh.cell_dofs.length[ci] = length(cell_dofs)

        @debug "Dofs for cell #$ci:\n\t$cell_dofs"
    end # cell loop
    return nextdof
end

#Use the variable name "face" here, but use for edge, vertex or cell aswell
function _add_dofs(cell_dofs, cell, dict, field_dim, nfacedofs, nextdof, faces::Function, sortface::Function)
    for face in faces(cell) # or vertices(), edges()
        key = sortface(face) # or sortedge(), sortedge()
        nextdof = get_or_create_dofs!(nextdof, field_dim, nfacedofs, cell_dofs, dict, key)
    end
    return nextdof
end

function matrixrotate!(matrix, ntimes::Int)
    for i in 1:ntimes
        matrix .= reverse(transpose!(matrix), dims=2)
    end
end

hexahedron_face_orintation(faceid::Int) = [false, true, false, false, true, true][faceid]

#Use the variable name "face" here, but use for edge, vertex or cell aswell
function add_face_dofs(cell_dofs, cell, dict, field_dim, nfacedofs, nextdof, faces::Function, sortface::Function)
    for (i, face) in enumerate(faces(cell)) # or vertices(), edges()
        key = sortface(face) # or sortedge(), sortedge()
        
        usedof = get!(dict, key) do 
            visited = false
            return nextdof, first(face)
        end
        
        if visited
            @debug println("      Creating next dof#$(nextdof)")
            nrotations = firstindex( (id) -> id == firstid, face) - 2
            nrotations = nrotations < 0 ? 3 : nrotations
            nrows = isqrt(nfacedofs)
            dofsmatrix = collect(CartesianIndices((nrows,nrows)))
            matrixrotate!(dofsmatrix, nrotations)
            append!(celldofs, dofsmatrix)
        else

        end

        nextdof = get_or_create_dofs!(nextdof, field_dim, nfacedofs, cell_dofs, dict, key)
    end
    return nextdof
end

"""
Add dofs to the cell_dofs vector, and return next available dof.
If dofs have already been created for the object (vertex, face, edge), then simply reuse them, otherwise create new dofs.
"""
function get_or_create_dofs!(nextdof::Int, field_dim::Int, nfacedofs::Int, cell_dofs::Vector{Int}, dict::Dict, key)

    visited = true
    usedof = get!(dict, key) do 
        @debug println("      Creating next dof#$(nextdof)")
        visited = false
        return nextdof
    end

    @debug visited && println("      Reusing dof#$(usedof)")

    for _ in 1:nfacedofs # or nedgedofs, nvertexdofs, ncelldofs
        for _ in 1:field_dim
            push!(cell_dofs, usedof)
            usedof += 1
        end
    end

    return visited ? nextdof : usedof
end

# TODO if not too slow it can replace the "Grid-version"
function _create_sparsity_pattern(dh::MixedDofHandler, sym::Bool)
    ncells = getncells(dh.grid)
    N::Int = 0
    for element_id = 1:ncells  # TODO check for correctness
        n = ndofs_per_cell(dh, element_id)
        N += sym ? div(n*(n+1), 2) : n^2
    end
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)

    cnt = 0
    for element_id in 1:ncells
        n = ndofs_per_cell(dh, element_id)
        global_dofs = zeros(Int, n)
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

@inline create_sparsity_pattern(dh::MixedDofHandler) = _create_sparsity_pattern(dh, false)


function find_field(fh::FieldHandler, field_name::Symbol)
    j = findfirst(i->i == field_name, getfieldnames(fh))
    j == 0 && error("did not find field $field_name")
    return j
end

# Calculate the offset to the first local dof of a field
function field_offset(fh::FieldHandler, field_name::Symbol)
    offset = 0
    for i in 1:find_field(fh, field_name)-1
        offset += getnbasefunctions(getfieldinterpolations(fh)[i])::Int * getfielddims(fh)[i]
    end
    return offset
end

find_field(dh::MixedDofHandler, field_name::Symbol) = find_field(first(dh.fieldhandlers), field_name)
field_offset(dh::MixedDofHandler, field_name::Symbol) = field_offset(first(dh.fieldhandlers), field_name)
getfieldinterpolation(dh::MixedDofHandler, field_idx::Int) = dh.fieldhandlers[1].fields[field_idx].interpolation
getfielddim(dh::MixedDofHandler, field_idx::Int) = dh.fieldhandlers[1].fields[field_idx].dim
getbcvalue(dh::MixedDofHandler, field_idx::Int) =  dh.fieldhandlers[1].bc_values[field_idx]
