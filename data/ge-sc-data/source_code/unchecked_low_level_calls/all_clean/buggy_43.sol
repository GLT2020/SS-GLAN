pragma solidity ^0.5.0;
contract EventMetadata {
  function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
  event MetadataSet(bytes metadata);
    function _setMetadata(bytes memory metadata) internal {
        emit MetadataSet(metadata);
    }
function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
}
contract Operated {
  function bug_unchk_send16() payable public{
      msg.sender.transfer(1 ether);}
  address private _operator;
  function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
  bool private _status;
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event OperatorUpdated(address operator, bool status);
    function _setOperator(address operator) internal {
        require(_operator != operator, "cannot set same operator");
        _operator = operator;
        emit OperatorUpdated(operator, hasActiveOperator());
    }
function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
    function _transferOperator(address operator) internal {
        require(_operator != address(0), "operator not set");
        _setOperator(operator);
    }
function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
    function _renounceOperator() internal {
        require(hasActiveOperator(), "only when operator active");
        _operator = address(0);
        _status = false;
        emit OperatorUpdated(address(0), false);
    }
function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
    function _activateOperator() internal {
        require(!hasActiveOperator(), "only when operator not active");
        _status = true;
        emit OperatorUpdated(_operator, true);
    }
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
    function _deactivateOperator() internal {
        require(hasActiveOperator(), "only when operator active");
        _status = false;
        emit OperatorUpdated(_operator, false);
    }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
    function getOperator() public view returns (address operator) {
        operator = _operator;
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function isOperator(address caller) public view returns (bool ok) {
        return (caller == getOperator());
    }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    function hasActiveOperator() public view returns (bool ok) {
        return _status;
    }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    function isActiveOperator(address caller) public view returns (bool ok) {
        return (isOperator(caller) && hasActiveOperator());
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
}
contract MultiHashWrapper {
    struct MultiHash {
        bytes32 hash;
        uint8 hashFunction;
        uint8 digestSize;
    }
    function _combineMultiHash(MultiHash memory multihash) internal pure returns (bytes memory) {
        bytes memory out = new bytes(34);
        out[0] = byte(multihash.hashFunction);
        out[1] = byte(multihash.digestSize);
        uint8 i;
        for (i = 0; i < 32; i++) {
          out[i+2] = multihash.hash[i];
        }
        return out;
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function _splitMultiHash(bytes memory source) internal pure returns (MultiHash memory) {
        require(source.length == 34, "length of source must be 34");
        uint8 hashFunction = uint8(source[0]);
        uint8 digestSize = uint8(source[1]);
        bytes32 hash;
        assembly {
          hash := mload(add(source, 34))
        }
        return (MultiHash({
          hashFunction: hashFunction,
          digestSize: digestSize,
          hash: hash
        }));
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
}
 interface iFactory {
     event InstanceCreated(address indexed instance, address indexed creator, string initABI, bytes initData);
     function create(bytes calldata initData) external returns (address instance);
     function createSalty(bytes calldata initData, bytes32 salt) external returns (address instance);
     function getInitSelector() external view returns (bytes4 initSelector);
     function getInstanceRegistry() external view returns (address instanceRegistry);
     function getTemplate() external view returns (address template);
     function getSaltyInstance(bytes calldata, bytes32 salt) external view returns (address instance);
     function getNextInstance(bytes calldata) external view returns (address instance);
     function getInstanceCreator(address instance) external view returns (address creator);
     function getInstanceType() external view returns (bytes4 instanceType);
     function getInstanceCount() external view returns (uint256 count);
     function getInstance(uint256 index) external view returns (address instance);
     function getInstances() external view returns (address[] memory instances);
     function getPaginatedInstances(uint256 startIndex, uint256 endIndex) external view returns (address[] memory instances);
 }
contract ProofHash is MultiHashWrapper {
  function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
  MultiHash private _proofHash;
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event ProofHashSet(address caller, bytes proofHash);
    function _setProofHash(bytes memory proofHash) internal {
        _proofHash = MultiHashWrapper._splitMultiHash(proofHash);
        emit ProofHashSet(msg.sender, proofHash);
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function getProofHash() public view returns (bytes memory proofHash) {
        proofHash = MultiHashWrapper._combineMultiHash(_proofHash);
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
}
contract Template {
  function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
  address private _factory;
    modifier initializeTemplate() {
        _factory = msg.sender;
        uint32 codeSize;
        assembly { codeSize := extcodesize(address) }
        require(codeSize == 0, "must be called within contract constructor");
        _;
    }
    function getCreator() public view returns (address creator) {
        creator = iFactory(_factory).getInstanceCreator(address(this));
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function isCreator(address caller) public view returns (bool ok) {
        ok = (caller == getCreator());
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function getFactory() public view returns (address factory) {
        factory = _factory;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
}
contract Post is ProofHash, Operated, EventMetadata, Template {
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event Initialized(address operator, bytes multihash, bytes metadata);
    function initialize(
        address operator,
        bytes memory multihash,
        bytes memory metadata
    ) public initializeTemplate() {
        if (multihash.length != 0) {
            ProofHash._setProofHash(multihash);
        }
        if (operator != address(0)) {
            Operated._setOperator(operator);
            Operated._activateOperator();
        }
        if (metadata.length != 0) {
            EventMetadata._setMetadata(metadata);
        }
        emit Initialized(operator, multihash, metadata);
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function setMetadata(bytes memory metadata) public {
        require(Template.isCreator(msg.sender) || Operated.isActiveOperator(msg.sender), "only active operator or creator");
        EventMetadata._setMetadata(metadata);
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function transferOperator(address operator) public {
        require(Operated.isActiveOperator(msg.sender), "only active operator");
        Operated._transferOperator(operator);
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function renounceOperator() public {
        require(Operated.isActiveOperator(msg.sender), "only active operator");
        Operated._renounceOperator();
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
}