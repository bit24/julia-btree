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

@propagate_inbounds function insert!(node::Node{K}, iData::K)::Union{Missing, Tuple{K, Node{K}}}  where K
    i = fgeq(node, iData)

    if i <= node.numData && node.data[i] == iData
        node.data[i] = iData
        return missing
    end
    
    if !isLeaf(node)
        ret = insert!(node.children[i], iData)
        
        if ismissing(ret)
            return missing
        end
        
        iData, iChild = ret
    end
    

    split = missing
    if node.numData == S
        rNode = Node{K}()
        m = S ÷ 2 + 1 # bias left

        if S & 1 == 0 # adjustments for even case
            if i == m # median                
                copy!(rNode.data, 1, node.data, m, node.numData)
                
                if !isLeaf(node)
                    copy!(rNode.children, 2, node.children, m+1, node.numData + 1)
                    rNode.children[1] = iChild
                end
                
                rNode.numData = node.numData - (m - 1)
                
                node.numData = m - 1
                
                return iData, rNode
            elseif i < m # if biased and left
                m -= 1
            end
        end
        
        copy!(rNode.data, 1, node.data, m+1, node.numData)
        
        if !isLeaf(node)
            copy!(rNode.children, 1, node.children, m+1, node.numData + 1)
        end
        
        rNode.numData = node.numData - m
        node.numData = m-1
        split = node.data[m]
        
        if i > m
            i -= m
            node = rNode
        end
    end

    shiftr!(node.data, i, node.numData)
    node.data[i] = iData
    
    # insert!(node.data, i, iData)
    
    # println(node.numData)
    if !isLeaf(node)
        shiftr!(node.children, i+1, node.numData + 1)
        node.children[i+1] = iChild
    end
    
    node.numData += 1
        
    return ismissing(split) ? missing : (split, rNode)
end

function insert!(tree::Tree{K}, iData::K) where K
    ret = insert!(tree.root, iData)
    
    if !ismissing(ret)
        nRoot = Node{K}()
        nRoot.children[1] = tree.root
        nRoot.data[1], nRoot.children[2] = ret
        nRoot.numData = 1
        tree.root = nRoot
    end
end

function push!(tree::Tree{K}, iData::K) where K
    insert!(tree, iData)
end

@propagate_inbounds function delete!(node::Node{K}, par::OptNode, pNodeI::Int, dData::K)::Union{Missing, Int, Node{K}} where K
    i = fgeq(node, dData)
    
    if !isLeaf(node)
        if i <= node.numData && node.data[i] == dData
            dNode = node.children[i]

            while !isLeaf(dNode)
                dNode = dNode.children[dNode.numData + 1]
            end
            rValue = dNode.data[dNode.numData]

            node.data[i] = rValue

            dData = rValue
        end
        
        i = delete!(node.children[i], node, i, dData)
        if ismissing(i)
            return missing
        end
    end
    
    if isLeaf(node)
        if i > node.numData || node.data[i] != dData
            return missing
        end
    end
    
    # by this point, assumes data i and children i+1 can be safely removed

    shiftl!(node.data, i+1, node.numData)
    if !isLeaf(node)
        shiftl!(node.children, i+2, node.numData+1)
    end
    
    node.numData -= 1
    
    if node.numData < S / 2
        if pNodeI == -1
            # root case: can't do anything about it
            
            if node.numData == 0
                return node.children[1]
            end
            
            return missing
        end
        
        if pNodeI != 1
            lSib = par.children[pNodeI-1]
            if lSib.numData > S / 2
                shiftr!(node.data, 1, node.numData)
                
                if !isLeaf(node)
                    shiftr!(node.children, 1, node.numData+1)
                    node.children[1] = lSib.children[lSib.numData+1]
                    lSib.children[lSib.numData+1] = missing
                end
                
                node.data[1] = par.data[pNodeI - 1]
                par.data[pNodeI - 1] = lSib.data[lSib.numData]
                
                node.numData += 1
                lSib.numData -= 1
                return missing
            end
        end
            
        if pNodeI != par.numData + 1
            rSib = par.children[pNodeI+1]
            if rSib.numData > S / 2
                node.data[node.numData+1] = par.data[pNodeI]
                par.data[pNodeI] = rSib.data[1]
                shiftl!(rSib.data, 2, rSib.numData)
                
                if !isLeaf(node)
                    node.children[node.numData+2]= rSib.children[1]
                    shiftl!(rSib.children, 2, rSib.numData+1)
                end
                
                node.numData += 1
                rSib.numData -= 1
                return missing
            end
        end
        
        if pNodeI != 1
            rNode = node
            node = par.children[pNodeI-1]
            pNodeI -= 1
        else
            rNode = par.children[pNodeI+1]
        end
        node.data[node.numData + 1] = par.data[pNodeI]

        copy!(node.data, node.numData + 2, rNode.data, 1, rNode.numData)
        
        copy!(node.children, node.numData + 2, rNode.children, 1, rNode.numData+1)
        
        node.numData += 1 + rNode.numData
                
        return pNodeI
    end
    return missing
end

function delete!(tree::Tree{K}, dData::K) where K
    ret = delete!(tree.root, missing, -1, dData)
    if !ismissing(ret)
        tree.root = ret
    end
    return nothing
end
