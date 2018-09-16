# this file defines iterators used for looping over a grid
struct UpdateFlags
    nodes::Bool
    coords::Bool
    celldofs::Bool
end

UpdateFlags(; nodes::Bool=true, coords::Bool=true, celldofs::Bool=true) =
    UpdateFlags(nodes, coords, celldofs)

"""
    CellIterator(grid::Grid)
    CellIterator(grid::DofHandler)

Return a `CellIterator` to conveniently loop over all the cells in a grid.

# Examples
```julia
for cell in CellIterator(grid)
    coords = getcoordinates(cell) # get the coordinates
    dofs = celldofs(cell)         # get the dofs for this cell
    reinit!(cv, cell)             # reinit! the FE-base with a CellIterator
end
```
"""
struct CellIterator{dim,N,T,M}
    flags::UpdateFlags
    grid::Grid{dim,T}
    current_cellid::ScalarWrapper{Int}
    nodes::Vector{Int}
    coords::Vector{Vec{dim,T}}
    dh::DofHandler{dim,T}
    celldofs::Vector{Int}
    cellset::Vector{Int}

    function CellIterator{dim,N,T,M}(dh::DofHandler{dim,T}, cellset::Vector{Int}, flags::UpdateFlags) where {dim,N,T,M}
        cell = ScalarWrapper(0)
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim,T}, N)
        n = ndofs_per_cell(dh, first(cellset))
        celldofs = zeros(Int, n)
        return new{dim,N,T,M}(flags, dh.grid, cell, nodes, coords, dh, celldofs, cellset)
    end

end

function CellIterator(dh::DofHandler{dim,T}, element::Element, flags::UpdateFlags=UpdateFlags()) where {dim,T}
    CellIterator{dim,nnodes(element),T,nfaces(element)}(dh, collect(get_elementcells(dh, element)), flags)
end
function CellIterator(dh::DofHandler{dim,T}, element::Element, cellset::Vector{Int}, flags::UpdateFlags=UpdateFlags()) where {dim,T}
    CellIterator{dim,nnodes(element),T,nfaces(element)}(dh, cellset, flags)
end

# iterator interface
function Base.iterate(ci::CellIterator, state = 1)
    if state > length(ci.cellset)
        return nothing
    else
        return (reinit!(ci, state), state+1)
    end
end
Base.length(ci::CellIterator)  = length(ci.cellset)

Base.IteratorSize(::Type{T})   where {T<:CellIterator} = Base.HasLength() # this is default in Base
Base.IteratorEltype(::Type{T}) where {T<:CellIterator} = Base.HasEltype() # this is default in Base
Base.eltype(::Type{T})         where {T<:CellIterator} = T

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
@inline nfaces(ci::CellIterator) = nfaces(eltype(ci.grid.cells))
@inline onboundary(ci::CellIterator, face::Int) = ci.grid.boundary_matrix[face, ci.current_cellid[]]
@inline cellid(ci::CellIterator) = ci.current_cellid[]
@inline celldofs!(v::Vector, ci::CellIterator) = celldofs!(v, ci.dh, ci.current_cellid[])
@inline celldofs(ci::CellIterator) = ci.celldofs

function reinit!(ci::CellIterator{dim,N}, i::Int) where {dim,N}
    ci.current_cellid[] = ci.cellset[i]
    nodeids = ci.grid.cells[ci.current_cellid[]].nodes

    @inbounds for j in 1:N
        nodeid = nodeids[j]
        ci.flags.nodes  && (ci.nodes[j] = nodeid)
        ci.flags.coords && (ci.coords[j] = ci.grid.nodes[nodeid].x)
    end
    if isdefined(ci, :dh) && ci.flags.celldofs # update celldofs
        celldofs!(ci.celldofs, ci)
    end
    return ci
end

@inline reinit!(cv::CellValues{dim,T}, ci::CellIterator{dim,N,T}) where {dim,N,T} = reinit!(cv, ci.coords)
@inline reinit!(fv::FaceValues{dim,T}, ci::CellIterator{dim,N,T}, face::Int) where {dim,N,T} = reinit!(fv, ci.coords, face)
