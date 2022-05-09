using DataStructures
# using StaticArrays
using BenchmarkTools
using Profile
using Random
import Base: insert!, delete!, haskey, push!, @propagate_inbounds
import DataStructures: AVLTreeNode, RBTreeNode, SplayTreeNode

const MEMMOVE = false
const BINSEARCH = true
const S = 64

mutable struct Node{K}
    data::Vector{K}
    children::Vector{Union{Missing, Node{K}}}
    numData::Int
end

# note: data can be changed to type Vector{Union{Missing, K}} too if isn't bits
# julia should call the correction function based on type

OptNode{K} = Union{Missing, Node{K}}

@inline Node{K}() where K = Node{K}(Vector{K}(undef, S), Vector{OptNode{K}}(missing, S + 1), 0)

@inline isLeaf(node::Node{K}) where K = ismissing(node.children[1])

mutable struct Tree{K}
    root::Node{K}
end

Tree{K}() where K = Tree{K}(Node{K}())

@inline function fgeq(node::Node{K}, d::K) where K
    if BINSEARCH
        l = 1
        h = node.numData + 1
        while l != h
            m = (l + h) >> 1
            if @inbounds node.data[m] < d
                l = m + 1
            else
                h = m
            end
        end
        return l
    else
        i = 1
        @inbounds while i <= node.numData && node.data[i] < d
            i += 1
        end
        return i
    end
end

# candidate for simd, unrolling, ccall, other optimizations
@inline function copy!(a::Vector{T}, sA::Int, b::Vector{T}, sB::Int, eB::Int) where T       
    if MEMMOVE
        unsafe_copyto!(a, sA, b, sB, eB-sB+1)
    else
        for i in 0:eB-sB
            @inbounds a[sA + i] = b[sB + i]
        end
    end
end

# remove references to enable garbage collection
@inline function copy!(a::Vector{Union{Missing, T}}, sA::Int, b::Vector{Union{Missing, T}}, sB::Int, eB::Int) where T       
    if MEMMOVE
        unsafe_copyto!(a, sA, b, sB, eB-sB+1)
    
        @inbounds @simd for i in sB:eB
            @inbounds b[i] = missing
        end
    else
        for i in 0:eB-sB
            @inbounds a[sA + i] = b[sB + i]
            @inbounds b[sB + i] = missing
        end
    end
end

@inline function shiftr!(a::Vector{Union{Missing, T}}, s::Int, e::Int) where T
    if MEMMOVE
        unsafe_copyto!(a, s+1, a, s, e-s+1)
        @inbounds a[s] = missing
    else
        for i in e:-1:s
            @inbounds a[i+1] = a[i]
        end
        @inbounds a[s] = missing
    end
end

@inline function shiftr!(a::Vector{T}, s::Int, e::Int) where T
    if MEMMOVE
        unsafe_copyto!(a, s+1, a, s, e-s+1)
    else
        for i in e:-1:s
            @inbounds a[i+1] = a[i]
        end
    end
end

@inline function shiftl!(a::Vector{Union{Missing, T}}, s::Int, e::Int) where T
    if MEMMOVE
        unsafe_copyto!(a, s-1, a, s, e-s+1)
        @inbounds a[e] = missing
    else
        for i in s:e
            @inbounds a[i-1] = a[i]
        end
        @inbounds a[e] = missing
    end
end

@inline function shiftl!(a::Vector{T}, s::Int, e::Int) where T
    if MEMMOVE
        unsafe_copyto!(a, s-1, a, s, e-s+1)
    else
        for i in s:e
            @inbounds a[i-1] = a[i]
        end
    end
end

@inline function findkey(node::Node{K}, d::K) where K
    while true
        i = fgeq(node, d)

        if i > node.numData
            return missing
        end

        if @inbounds node.data[i] == d
           @inbounds return node.data[i]
        end

        if isLeaf(node)
            return missing
        end
        
        node = node.children[i]
    end
end

@inline function haskey(tree::Tree{K}, key::K) where K
    return !ismissing(findkey(tree.root, key))
end

#unoptimized implemenentation
function toList!(l::Vector{K}, node::Node{K}) where K
    for i = 1:node.numData
        if !isLeaf(node)
            toList!(l, node.children[i])
        end
        append!(l, node.data[i])
    end
    if !isLeaf(node)
        toList!(l, node.children[node.numData+1])
    end
end

function toList!(l::Vector{K}, tree::Tree{K}) where K
    toList!(l, tree.root)
end

# unoptimized implemenentation
function lower_bound(node::Node{K}, k::K) where K
    i = fgeq(node, k)
    
    if i > node.numData
        return missing
    end
    
    if isLeaf(node)
        return node.data[i]
    else
        v = lower_bound(node.children[i], k)
        return ismissing(v) ? node.data[i] : v
    end
end

function lower_bound(tree::Tree{K}, k::K) where K
    return lower_bound(tree.root, k)
end