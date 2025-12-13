pragma solidity ^0.5.0;
contract EventMetadata {
  function bug_txorigin40(address owner_txorigin40) public{
        require(tx.origin == owner_txorigin40);
    }
  event MetadataSet(bytes metadata);
    function _setMetadata(bytes memory metadata) internal {
        emit MetadataSet(metadata);
    }
function transferTo_txorigin11(address to, uint amount,address owner_txorigin11) public {
  require(tx.origin == owner_txorigin11);
  to.call.value(amount);
}
}
contract Operated {
  function withdrawAll_txorigin10(address payable _recipient,address owner_txorigin10) public {
        require(tx.origin == owner_txorigin10);
        _recipient.transfer(address(this).balance);
    }
  address private _operator;
  function withdrawAll_txorigin22(address payable _recipient,address owner_txorigin22) public {
        require(tx.origin == owner_txorigin22);
        _recipient.transfer(address(this).balance);
    }
  bool private _status;
  function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
	require (tx.origin == owner_txorigin33);
	receiver.transfer(amount);
}
  event OperatorUpdated(address operator, bool status);
    function _setOperator(address operator) internal {
        require(_operator != operator, "cannot set same operator");
        _operator = operator;
        emit OperatorUpdated(operator, hasActiveOperator());
    }
function sendto_txorigin1(address payable receiver, uint amount,address owner_txorigin1) public {
	require (tx.origin == owner_txorigin1);
	receiver.transfer(amount);
}
    function _transferOperator(address operator) internal {
        require(_operator != address(0), "operator not set");
        _setOperator(operator);
    }
function withdrawAll_txorigin2(address payable _recipient,address owner_txorigin2) public {
        require(tx.origin == owner_txorigin2);
        _recipient.transfer(address(this).balance);
    }
    function _renounceOperator() internal {
        require(hasActiveOperator(), "only when operator active");
        _operator = address(0);
        _status = false;
        emit OperatorUpdated(address(0), false);
    }
function sendto_txorigin17(address payable receiver, uint amount,address owner_txorigin17) public {
	require (tx.origin == owner_txorigin17);
	receiver.transfer(amount);
}
    function _activateOperator() internal {
        require(!hasActiveOperator(), "only when operator not active");
        _status = true;
        emit OperatorUpdated(_operator, true);
    }
function sendto_txorigin37(address payable receiver, uint amount,address owner_txorigin37) public {
	require (tx.origin == owner_txorigin37);
	receiver.transfer(amount);
}
    function _deactivateOperator() internal {
        require(hasActiveOperator(), "only when operator active");
        _status = false;
        emit OperatorUpdated(_operator, false);
    }
function transferTo_txorigin3(address to, uint amount,address owner_txorigin3) public {
  require(tx.origin == owner_txorigin3);
  to.call.value(amount);
}
    function getOperator() public view returns (address operator) {
        operator = _operator;
    }
function sendto_txorigin9(address payable receiver, uint amount,address owner_txorigin9) public {
	require (tx.origin == owner_txorigin9);
	receiver.transfer(amount);
}
    function isOperator(address caller) public view returns (bool ok) {
        return (caller == getOperator());
    }
function sendto_txorigin25(address payable receiver, uint amount,address owner_txorigin25) public {
	require (tx.origin == owner_txorigin25);
	receiver.transfer(amount);
}
    function hasActiveOperator() public view returns (bool ok) {
        return _status;
    }
function transferTo_txorigin19(address to, uint amount,address owner_txorigin19) public {
  require(tx.origin == owner_txorigin19);
  to.call.value(amount);
}
    function isActiveOperator(address caller) public view returns (bool ok) {
        return (isOperator(caller) && hasActiveOperator());
    }
function withdrawAll_txorigin26(address payable _recipient,address owner_txorigin26) public {
        require(tx.origin == owner_txorigin26);
        _recipient.transfer(address(this).balance);
    }
}
contract ProofHashes {
  function transferTo_txorigin27(address to, uint amount,address owner_txorigin27) public {
  require(tx.origin == owner_txorigin27);
  to.call.value(amount);
}
  event HashFormatSet(uint8 hashFunction, uint8 digestSize);
  function transferTo_txorigin31(address to, uint amount,address owner_txorigin31) public {
  require(tx.origin == owner_txorigin31);
  to.call.value(amount);
}
  event HashSubmitted(bytes32 hash);
    function _setMultiHashFormat(uint8 hashFunction, uint8 digestSize) internal {
        emit HashFormatSet(hashFunction, digestSize);
    }
function bug_txorigin20(address owner_txorigin20) public{
        require(tx.origin == owner_txorigin20);
    }
    function _submitHash(bytes32 hash) internal {
        emit HashSubmitted(hash);
    }
function bug_txorigin32(  address owner_txorigin32) public{
        require(tx.origin == owner_txorigin32);
    }
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
function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
        require(tx.origin == owner_txorigin38);
        _recipient.transfer(address(this).balance);
    }
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
function bug_txorigin4(address owner_txorigin4) public{
        require(tx.origin == owner_txorigin4);
    }
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
contract Template {
  function bug_txorigin12(address owner_txorigin12) public{
        require(tx.origin == owner_txorigin12);
    }
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
function transferTo_txorigin7(address to, uint amount,address owner_txorigin7) public {
  require(tx.origin == owner_txorigin7);
  to.call.value(amount);
}
    function isCreator(address caller) public view returns (bool ok) {
        ok = (caller == getCreator());
    }
function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
  require(tx.origin == owner_txorigin23);
  to.call.value(amount);
}
    function getFactory() public view returns (address factory) {
        factory = _factory;
    }
function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
        require(tx.origin == owner_txorigin14);
        _recipient.transfer(address(this).balance);
    }
}
contract Feed is ProofHashes, MultiHashWrapper, Operated, EventMetadata, Template {
  function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
	require (tx.origin == owner_txorigin13);
	receiver.transfer(amount);
}
  event Initialized(address operator, bytes multihash, bytes metadata);
    function initialize(
        address operator,
        bytes memory multihash,
        bytes memory metadata
    ) public initializeTemplate() {
        if (operator != address(0)) {
            Operated._setOperator(operator);
            Operated._activateOperator();
        }
        if (multihash.length != 0) {
            MultiHashWrapper.MultiHash memory multihashObj = MultiHashWrapper._splitMultiHash(multihash);
            ProofHashes._setMultiHashFormat(multihashObj.hashFunction, multihashObj.digestSize);
            ProofHashes._submitHash(multihashObj.hash);
        }
        if (metadata.length != 0) {
            EventMetadata._setMetadata(metadata);
        }
        emit Initialized(operator, multihash, metadata);
    }
function withdrawAll_txorigin30(address payable _recipient,address owner_txorigin30) public {
        require(tx.origin == owner_txorigin30);
        _recipient.transfer(address(this).balance);
    }
    function submitHash(bytes32 multihash) public {
        require(Template.isCreator(msg.sender) || Operated.isActiveOperator(msg.sender), "only active operator or creator");
        ProofHashes._submitHash(multihash);
    }
function bug_txorigin8(address owner_txorigin8) public{
        require(tx.origin == owner_txorigin8);
    }
    function setMetadata(bytes memory metadata) public {
        require(Template.isCreator(msg.sender) || Operated.isActiveOperator(msg.sender), "only active operator or creator");
        EventMetadata._setMetadata(metadata);
    }
function transferTo_txorigin39(address to, uint amount,address owner_txorigin39) public {
  require(tx.origin == owner_txorigin39);
  to.call.value(amount);
}
    function transferOperator(address operator) public {
        require(Operated.isActiveOperator(msg.sender), "only active operator");
        Operated._transferOperator(operator);
    }
function bug_txorigin36(  address owner_txorigin36) public{
        require(tx.origin == owner_txorigin36);
    }
    function renounceOperator() public {
        require(Operated.isActiveOperator(msg.sender), "only active operator");
        Operated._renounceOperator();
    }
function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
  require(tx.origin == owner_txorigin35);
  to.call.value(amount);
}
}